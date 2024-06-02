"""Test the trained models on the game and save a video file as output"""

import sys
import torch
import pygame
from pygame_screen_record import ScreenRecorder
from pygame_screen_record.ScreenRecorder import cleanup

sys.path.append('./')
from source.game.flappy_bird import FlappyBird
from source.utils.process_image import pre_processing

def test(test_id):
    torch.manual_seed(22)
    torch.cuda.manual_seed(22)
    device = "cuda"
    IMAGE_SIZE = 84
    
    with torch.no_grad():
        model = torch.load(f'./model_ckpts/{test_id}_model.pt')
        
        game_state = FlappyBird()
        image, reward, terminal = game_state.next_frame(0, disp_score=True)
        image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], 
                               IMAGE_SIZE, IMAGE_SIZE)
        
        image = torch.from_numpy(image).to(device)
        
        state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

        recorder = ScreenRecorder(30) # pass your desired fps
        recorder.start_rec() # start recording
        try:
            while not terminal:
                # Get output from the neural network
                with torch.no_grad():
                    prediction = model(state)[0]
                
                action = torch.argmax(prediction).item()

                next_image, reward, terminal = game_state.next_frame(action, disp_score=True)
                next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], 
                                    IMAGE_SIZE, IMAGE_SIZE)
        
                next_image = torch.from_numpy(next_image).to(device)
        
                next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]

                state = next_state
        finally:
            recorder.stop_rec()	# stop recording
            recorder.save_recording(f"./videos/{test_id}.avi") # saves the last recording
            cleanup()
            pygame.quit()
        

if __name__ == "__main__":
    
    test(test_id=0)
