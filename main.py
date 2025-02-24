from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color, Line
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.core.image import Image

from tools import object_crop, preprocess_image
import load_model


class PaintWidget(Widget):

    timeout_event = None

    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 0)
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=10.0)
        if self.timeout_event:
            Clock.unschedule(self.timeout_event)

    def on_touch_move(self, touch):
        touch.ud["line"].points += [touch.x, touch.y]

    def on_touch_up(self, touch):
        self.timeout_event = Clock.schedule_once(self.save_image, 3)

    def save_image(self, dt):
        # print("Se ha completado una letra o palabra")
        # self.parent.ids.text_input.text = "nueva palabra"
        self.export_to_png("trazo.jpg")
        img = object_crop("trazo.jpg")
        img = preprocess_image(img)
        rec_char = load_model.predict_digit(img)
        self.parent.ids.text_input.text = rec_char
        self.clear_canvas()

    def clear_canvas(self):
        self.canvas.clear()


class Controller(BoxLayout):
    pass


class HandWriteApp(App):
    pass


if __name__ == "__main__":
    HandWriteApp().run()
