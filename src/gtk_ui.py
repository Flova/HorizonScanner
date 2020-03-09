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
        self.base_path = os.path.dirname(__file__)

        # Load gtk builder
        self.builder = Gtk.Builder()
        self.builder.add_from_file(os.path.join(self.base_path, "ui/gui_v_1.glade"))

        # Define current interface state
        self.state = {
            'rewind': False,
            'play': True,
            'record': False,
            'auto': False,
        }
        self.previous_state = self.state.copy()


        # Load config
        config_path = os.path.join(self.base_path, "../config/config.yaml")
        config_path = os.path.realpath(config_path)
        if not os.path.exists(config_path):
            print("No config file specified, see the 'example.config.yaml' in 'config' and save your version as 'config.yaml'!")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load boat detector
        self.boat_detector = BoatDetector(self.config)

        #Create image buffer for rewind
        self.buffer = []
        self.buffer_size = self.config['gui']['image_buffer_size']
        self.rewind_buffer = self.buffer.copy()
        self.rewind_speed = self.config['gui']['rewind_speed']
        self._rewind_speed_counter = 0

        # Set video source
        self.video_source = self.config['video_source']

        # Placeholder for the video recorder object
        self.video_recorder = None
        # Image placeholders
        self.image_shape = None
        self.image = None

        # Init capture device / image loader
        self.cap = cv2.VideoCapture(self.video_source)
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)

        # Load splash image
        self.pixbuf = GdkPixbuf.Pixbuf().new_from_file(os.path.join(self.base_path, 'ui/images/splash_screen.jpg'))

        # Add image canvas
        self.image_canvas = Gtk.Image().new_from_pixbuf(self.pixbuf)
        self.parent = self.builder.get_object("main_window_image_canvas_container")
        self.parent.add(self.image_canvas)

        # Create candidate ui list
        self.candidate_liststore = Gtk.ListStore(GdkPixbuf.Pixbuf, str)
        iconview = self.builder.get_object("candiate_gallery")
        iconview.set_model(self.candidate_liststore)
        iconview.set_pixbuf_column(0)
        iconview.set_text_column(1)

        # Connect all signals to corresponding callbacks
        self.builder.connect_signals(self)

        # Set true if threads should shutdown
        self.shutdown = False

        # Show window
        window = self.builder.get_object("main_window")
        window.show_all()

        # Start image loading runtime
        self.video_thread = Thread(target=self.image_get_runtime)
        self.video_thread.daemon = True
        self.video_thread.start()

        # Start detector runtime
        self.detector_thread = Thread(target=self.detector_runtime)
        self.detector_thread.daemon = True
        self.detector_thread.start()

        # Gtk main thread
        Gtk.main()

    def add_to_buffer(self, image):
        """
        Adds image to the image buffer queue
        """
        self._rewind_speed_counter = (self._rewind_speed_counter + 1) %  self.rewind_speed
        if self._rewind_speed_counter == 0:
            if len(self.buffer) > self.buffer_size * self.frame_rate:
                del self.buffer[0]
            self.buffer.append(image)

    def display_image(self, image):
        """
        Sends the image to the GTK Main thread while it is in idle mode
        """
        # Sacle image so the UI has the same size, no matter the input resolution
        image_height = 800
        image_width = int((self.image_shape[1]/self.image_shape[0])*image_height)
        image = cv2.resize(image.astype(np.uint8), (image_width, image_height))
        # Send image
        GLib.idle_add(self.update_image, image)

    def display_candiates(self, roi, candidates, candidate_movement):
        """
        Sends the detected candidates to the GTK Main thread while it is in idle mode
        """
        for index, candidate in enumerate(candidates):
            # Check if the candidate is shown in the window
            if not index in candidate_movement.values():
                # Get pixel values from bounding box
                candidate = roi[:, candidate[0]: candidate[1], :]
                # Send
                GLib.idle_add(self.add_candidate, candidate.astype(np.uint8))

    def detector_runtime(self):
        """
        Runs an instance of the BoatDetector
        """
        old_candidates = []
        while not self.shutdown:
            # Check if detector is activated
            if self.state['auto']:
                # Run detection on frame
                image = self.image.copy()
                # Run the detector on the image
                valid, roi, _, _, _, candidates  = self.boat_detector.analyse_image(image, roi_height=30, history=True)
                # Check if a valid result could be determined
                if valid:
                    # Get the pixel values for each bounding box
                    rendered_candidates = self.boat_detector.render_candiates(roi, candidates)
                    # Check if the candidate occurred before
                    candidate_movement = self.boat_detector.relocate_candidates(rendered_candidates, old_candidates)
                    # Send the candidates to the main thread
                    self.display_candiates(roi, candidates, candidate_movement)
                    old_candidates = rendered_candidates.copy()
            else:
                # Idle
                time.sleep(0.5)

    def image_get_runtime(self):
        """
        Acquires new images to display them if the correct mode is set.
        It also handles the rewind and record function.
        """
        while not self.shutdown:
            # Start timing for constant framerate
            starttime = time.time()
            # Read image
            ret, framebuffer = self.cap.read()
            # Close if video source is closed
            if not ret or framebuffer is None:
                print("Video source closed")
                self.on_close()
            # Get frame shape
            self.image_shape = framebuffer.shape
            # Save frame in the buffer
            self.add_to_buffer(framebuffer)
            # Set image
            self.image = framebuffer

            # Normal mode and not paused
            if not self.state['rewind'] and self.state['play']:
                # Display current frame
                self.display_image(framebuffer)

            # Rewind mode
            if self.state['rewind']:
                # Check if the rewind mode is freshly entered
                if not self.previous_state['rewind']:
                    # Copy the rewind buffer, so the current buffer runs unaffected
                    self.rewind_buffer = self.buffer.copy()
                # Check if buffer not empty
                if not len(self.rewind_buffer) == 0:
                    # Check paused
                    if self.state['play']:
                        # Display image
                        self.display_image(self.rewind_buffer[-1])
                        # Remove image from buffer
                        del self.rewind_buffer[-1]
                else:
                    # Reset state
                    del self.rewind_buffer
                    self.state['rewind'] = False


            # Recording mode
            if self.state['record']:
                # Check if the recorder needs to be crated (first recorded frame)
                if self.video_recorder is None:
                    # Get path where the video should be saved (TODO config)
                    video_output_path = os.path.realpath(
                        os.path.join(
                            __file__,
                            "../../data/",
                            time.strftime("%Y%m%d-%H%M%S") + ".mp4"))
                    # Set codec
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                    # Create video recorder
                    self.video_recorder = cv2.VideoWriter(
                        video_output_path,
                        fourcc,
                        self.frame_rate,
                        (self.image_shape[1],self.image_shape[0]))
                # Write frame to file
                self.video_recorder.write(framebuffer)
            # Stop video recording if we leave the recording state
            if not self.state['record'] and self.previous_state['record']:
                self.video_recorder.release()
                del self.video_recorder

            self.previous_state = self.state.copy()

            # Sleep for constant frame rate
            time_delta = time.time() - starttime
            sleep_time = max(1/float(self.frame_rate) - time_delta, 0)
            time.sleep(sleep_time)

    def on_main_window_destroy(self, *args):
        """
        Handle main window close event
        """
        self.on_close()

    def on_close(self):
        self.shutdown = True
        self.video_thread.join()
        self.detector_thread.join()
        Gtk.main_quit()

    def main_window_play_pause(self, *args):
        """
        Callback to switch the play state
        """
        play_button = self.builder.get_object("play_pause_button")
        self.state['play'] = not self.state['play']
        if self.state['play']:
            play_button.props.stock_id = Gtk.STOCK_MEDIA_PAUSE
        else:
            play_button.props.stock_id = Gtk.STOCK_MEDIA_PLAY

    def main_window_rewind(self, *args):
        """
        Callback to switch rewind state
        """
        self.state['rewind'] = True

    def main_window_record_toggle(self, button):
        """
        Callback to switch record state
        """
        self.state['record'] = button.get_active()

    def main_window_auto_toggled(self, button):
        """
        Callback to activate/deactivate the boat detection
        """
        # Change state
        self.state['auto'] = button.get_active()
        # Show/hide window
        candidate_window = self.builder.get_object("candidate_window")
        candidate_window.set_default_size(500, 500)
        if self.state['auto']:
            candidate_window.show()
        else:
            candidate_window.hide()

    def _cv_image_to_pixbuf(self, image):
        """
        Convert cv image (numpy array) to a gtk pixelbuffer
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return GdkPixbuf.Pixbuf().new_from_data(image.tostring(),
                                                0,
                                                False,
                                                8,
                                                image.shape[1],
                                                image.shape[0],
                                                image.shape[1]*3)

    def update_image(self, image):
        """
        Draws cv image on image canvas
        """
        pixel_buf = self._cv_image_to_pixbuf(image)
        self.image_canvas.set_from_pixbuf(pixel_buf)
        self.image_canvas.show()

    def add_candidate(self, candidate):
        """
        Appends queue of shown candidates in the candidate window
        """
        # Resize candidate to the correct thumbnail size
        # Clip max width
        max_width = 200
        if candidate.shape[1] * 2 > max_width:
            print("Too big")
            candidate = cv2.resize(candidate, (0,0), fx=max_width/(candidate.shape[1] * 2), fy=max_width/(candidate.shape[1] * 2))
        else:
            candidate = cv2.resize(candidate, (0,0), fx=2, fy=2)
        # Convert candidates
        pixbuf = self._cv_image_to_pixbuf(candidate)
        # Insert it at the front of the list
        self.candidate_liststore.insert(0, [pixbuf, ""])
        # Delete last element if needed
        if len(self.candidate_liststore) > self.config['gui']['candidate_list_length']:
            del self.candidate_liststore[-1]

if __name__ == "__main__":
    UserInterface()
