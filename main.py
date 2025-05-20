import sys
from datetime import datetime
from pathlib import Path

import glfw
import imgui
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
from PIL import Image

from pixel_sorter.pixel_sorter import PixelSorter

class GLImage:
    def __init__(self, pil_image: Image.Image):
        self.image = pil_image.convert("RGBA")
        self.texture_id = None
        self.width, self.height = self.image.size
        self.upload_texture()

    def upload_texture(self):
        data = self.image.tobytes("raw", "RGBA", 0, -1)

        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            self.width,
            self.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            data,
        )

    def scale_to_fit(self, max_width, max_height):
        aspect = self.width / self.height
        new_width = min(self.width, max_width)
        new_height = new_width / aspect

        if new_height > max_height:
            new_height = max_height
            new_width = new_height * aspect

        return int(new_width), int(new_height)

    def free_texture(self):
        if self.texture_id:
            gl.glDeleteTextures([self.texture_id])
            self.texture_id = None


def render_image(gl_image: GLImage, name: str = "Image", fit: bool = True):
    imgui.begin(name)
    available_w, available_h = imgui.get_content_region_available()
    if fit:
        w, h = gl_image.scale_to_fit(available_w, available_h)
    else:
        w, h = gl_image.width, gl_image.height
    imgui.image(gl_image.texture_id, w, h)
    imgui.end()

def get_image_paths():
    paths = []
    for extension in [".png", ".jpg", ".jpeg"]:
        paths.extend(Path("data", "input").glob(f"**/*{extension}"))

    paths = [str(p) for p in paths]
    return paths
        

def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    pixel_sorter: PixelSorter | None = None
    pixel_sorter_modes = ["R", "G", "B", "luminance", "hue"]
    pixel_sorter_directions = [
        "left-to-right",
        "right-to-left",
        "top-to-bottom",
        "bottom-to-top",
    ]
    pixel_sorter_image_paths = get_image_paths()

    mask_threshold_value: float = 20.0
    selected_image_index: int = 0
    selected_mode_index: int = 0
    selected_direction_index: int = 0

    original_gl_image: GLImage | None = None
    mask_gl_image: GLImage | None = None
    sorted_gl_image: GLImage | None = None

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):
                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit",
                    "Cmd+Q",
                    False,
                    True,
                )

                if clicked_quit:
                    sys.exit(0)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.begin("Options")
        
        selected_image_changed, selected_image_index = imgui.combo(
            "Select Image",
            selected_image_index,
            pixel_sorter_image_paths,
        )
        if selected_image_changed:
            input_path = pixel_sorter_image_paths[selected_image_index]
            pixel_sorter = PixelSorter(input_path, mask_threshold_value)
            if original_gl_image:
                original_gl_image.free_texture()
            if mask_gl_image:
                mask_gl_image.free_texture()

            original_gl_image = GLImage(pixel_sorter.pil_image)
            mask_gl_image = GLImage(pixel_sorter.mask)

        if imgui.button("Refresh list"):
            pixel_sorter_image_paths = get_image_paths()

        mask_threshold_changed, mask_threshold_value = imgui.drag_float(
            "Mask threshold",
            mask_threshold_value,
            1,
            0,
            255,
        )

        _, selected_mode_index = imgui.combo(
            "Sort mode",
            selected_mode_index,
            pixel_sorter_modes,
        )

        _, selected_direction_index = imgui.combo(
            "Sort direction",
            selected_direction_index,
            pixel_sorter_directions,
        )

        if imgui.button("Sort"):
            if sorted_gl_image:
                sorted_gl_image.free_texture()
            pixel_sorter_mode = pixel_sorter_modes[selected_mode_index]
            pixel_sorter_direction = pixel_sorter_directions[selected_direction_index]

            pixel_sorter.sort(pixel_sorter_mode, pixel_sorter_direction)
            sorted_gl_image = GLImage(pixel_sorter.sorted_image)

        if imgui.button("Save"):
            output_path = Path("data/output", f"{datetime.now().timestamp()}.png")
            pixel_sorter.save(output_path)

        imgui.end()

        if mask_threshold_changed and mask_gl_image:
            pixel_sorter.set_mask_threshold(mask_threshold_value)
            mask_gl_image.free_texture()
            mask_gl_image = GLImage(pixel_sorter.mask)

        if pixel_sorter:
            render_image(original_gl_image, "Original")
            render_image(mask_gl_image, "Mask")

        if sorted_gl_image:
            render_image(sorted_gl_image, "Sorted")

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1280, 720
    window_name = "Pixel sorter"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


if __name__ == "__main__":
    main()


    print(get_image_files())