import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GdkPixbuf, Gdk, GLib
from threading import Thread
from main import BoatDetector
import numpy as np
import yaml
import time
import cv2
import os

class UserInterface(object):
    def __init__(self):
        builder = Gtk.Builder()
        builder.add_from_file("ui/gui_v_1.glade")

        playback_state = {
            'rewind': False,
            'play': True,
            'record': False,
            'auto': False,
        }

        config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")

        config_path = os.path.realpath(config_path)

        if not os.path.exists(config_path):
            print("No config file specified, see the 'example.config.yaml' in 'config' and save your version as 'config.yaml'!")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        bt = BoatDetector(config)

        self.config = config

        self.video_manager = VideoManager(playback_state, bt, config)
        self._mwh = MainWindowHandler(builder, playback_state, config)

        self.video_manager.set_image_callback(self._mwh.update_image)
        self.video_manager.set_candidate_callback(self._mwh.add_candidate)

        window = builder.get_object("main_window")
        window.show_all()

        thread = Thread(target=self.video_manager.image_get_runtime)
        thread.daemon = True
        thread.start()

        thread = Thread(target=self.video_manager.detector_runtime)
        thread.daemon = True
        thread.start()

        Gtk.main()


class VideoManager(object):
    def __init__(self, state, boat_detector, config):
        self.buffer = []
        self.buffer_size = config['gui']['image_buffer_size']
        self.rewind_buffer = self.buffer.copy()

        self.boat_detector = boat_detector
        self.config = config

        self.rewind_speed = config['gui']['rewind_speed']
        self._rewind_speed_counter = 0

        self.video_source = self.config['video_source']

        self.display_image_callback = None
        self.display_candidate_callback = None

        self.video_recorder = None

        self.state = state
        self.previous_state = state.copy()

        self.image_shape = None
        self.image = None

        self.cap = cv2.VideoCapture(self.video_source)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

    def add_to_buffer(self, image):
        self._rewind_speed_counter = (self._rewind_speed_counter + 1) %  self.rewind_speed
        if self._rewind_speed_counter == 0:
            if len(self.buffer) > self.buffer_size * self.frame_rate:
                del self.buffer[0]
            self.buffer.append(image)

    def display_image(self, image):
        image_height = 800
        image_width = int((self.image_shape[1]/self.image_shape[0])*image_height)
        image = cv2.resize(image.astype(np.uint8), (image_width, image_height))
        GLib.idle_add(self.display_image_callback, image)

    def display_candiates(self, roi, candidates, candidate_movement):
        for index, candidate in enumerate(candidates):
            if not index in candidate_movement.values():
                candidate = roi[:, candidate[0]: candidate[1], :]
                GLib.idle_add(self.set_candidate_callback, candidate.astype(np.uint8))

    def set_image_callback(self, callback):
        self.display_image_callback = callback

    def set_candidate_callback(self, callback):
        self.set_candidate_callback = callback

    def detector_runtime(self):
        old_candidates = []
        while True:
            if self.state['auto']:
                # Run detection on frame
                image = self.image.copy()
                valid, roi, _, _, _, candidates  = self.boat_detector.analyse_image(image, roi_height=30, history=True)
                if valid:
                    rendered_candidates = self.boat_detector.render_candiates(roi, candidates)

                    candidate_movement = self.boat_detector.relocate_candidates(rendered_candidates, old_candidates)

                    self.display_candiates(roi, candidates, candidate_movement)

                    old_candidates = rendered_candidates.copy()
            else:
                time.sleep(0.5)

    def image_get_runtime(self):
        while True:
            starttime = time.time()
            ret, framebuffer = self.cap.read()

            if not ret:
                print("Video source closed")
                Gtk.main_quit()

            self.image_shape = framebuffer.shape

            self.add_to_buffer(framebuffer)

            self.image = framebuffer

            # Normal
            if not self.state['rewind'] and self.state['play']:
                self.display_image(framebuffer)

            # Rewind
            if self.state['rewind']:
                if not self.previous_state['rewind']:
                    self.rewind_buffer = self.buffer.copy()
                else:
                    if not len(self.rewind_buffer) == 0:
                        if self.state['play']:
                            self.display_image(self.rewind_buffer[-1])
                            del self.rewind_buffer[-1]
                    else:
                        del self.rewind_buffer
                        self.state['rewind'] = False


            # Recording
            if self.state['record']:
                if self.video_recorder is None:
                    video_output_path = os.path.realpath(
                        os.path.join(
                            __file__,
                            "../../data/",
                            time.strftime("%Y%m%d-%H%M%S") + ".mp4"))
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    self.video_recorder = cv2.VideoWriter(
                        video_output_path,
                        fourcc,
                        self.frame_rate,
                        (self.image_shape[1],self.image_shape[0]))
                self.video_recorder.write(framebuffer)

            if not self.state['record'] and self.previous_state['record']:
                self.video_recorder.release()
                del self.video_recorder

            self.previous_state = self.state.copy()

            time_delta = time.time() - starttime
            sleep_time = max(1/float(self.frame_rate) - time_delta, 0)
            time.sleep(sleep_time)


class MainWindowHandler(object):
        def __init__(self, builder, state, config):
            self.builder = builder

            self.state = state
            self.config = config

            # Load splash image and init image object
            self.pixbuf = GdkPixbuf.Pixbuf().new_from_file('ui/images/splash_screen.jpg')

            self.image_canvas = Gtk.Image().new_from_pixbuf(self.pixbuf)
            self.parent = self.builder.get_object("main_window_image_canvas_container")
            self.parent.add(self.image_canvas)

            self.candidate_liststore = Gtk.ListStore(GdkPixbuf.Pixbuf, str)
            iconview = self.builder.get_object("candiate_gallery")
            iconview.set_model(self.candidate_liststore)
            iconview.set_pixbuf_column(0)
            iconview.set_text_column(1)

            self.builder.connect_signals(self)

        def on_main_window_destroy(self, *args):
            Gtk.main_quit()

        def main_window_play_pause(self, *args):
            play_button = self.builder.get_object("play_pause_button")
            self.state['play'] = not self.state['play']
            if self.state['play']:
                play_button.props.stock_id = Gtk.STOCK_MEDIA_PAUSE
            else:
                play_button.props.stock_id = Gtk.STOCK_MEDIA_PLAY

        def main_window_rewind(self, *args):
            self.state['rewind'] = True

        def main_window_record_toggle(self, button):
            self.state['record'] = button.get_active()

        def main_window_auto_toggled(self, button):
            self.state['auto'] = button.get_active()
            candidate_window = self.builder.get_object("candidate_window")
            candidate_window.set_default_size(500, 500)
            if self.state['auto']:
                candidate_window.show()
            else:
                candidate_window.hide()

        def _cv_image_to_pixbuf(self, image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return GdkPixbuf.Pixbuf().new_from_data(image.tostring(),
                                                    0,
                                                    False,
                                                    8,
                                                    image.shape[1],
                                                    image.shape[0],
                                                    image.shape[1]*3)

        def update_image(self, image):
            pixel_buf = self._cv_image_to_pixbuf(image)
            self.image_canvas.set_from_pixbuf(pixel_buf)
            self.image_canvas.show()

        def add_candidate(self, candidate):
            candidate = cv2.resize(candidate, (0,0), fx=2, fy=2)
            pixbuf = self._cv_image_to_pixbuf(candidate)
            self.candidate_liststore.insert(0, [pixbuf, ""])
            if len(self.candidate_liststore) > self.config['gui']['candidate_list_length']:
                del self.candidate_liststore[-1]





if __name__ == "__main__":
    UserInterface()
