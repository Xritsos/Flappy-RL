"""Test the trained models on the game and save a video file as output"""

import sys
import torch
import pygame
from pygame_screen_record import ScreenRecorder
from pygame_screen_record.ScreenRecorder import cleanup

sys.path.append('./')
from source.game import wrapped_flappy_bird as game
from source.utils import Image 


def test(test_id):
    torch.manual_seed(22)
    device = "cuda"
    
    with torch.no_grad():
        model = torch.load(f'./model_ckpts/{test_id}_model.pt')
        
        game_state = game.GameState()
        number_of_actions = 2
        
        action = torch.zeros([number_of_actions], dtype=torch.float32).to(device)
        action[0] = 1
        
        image_data, reward, terminal = game_state.frame_step(action)
        image_data = Image.resize_and_bgr2gray(image_data)
        image_data = Image.image_to_tensor(image_data)
        image_data = image_data.to(device)
        state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

        recorder = ScreenRecorder(30) # pass your desired fps
        recorder.start_rec() # start recording
        try:
            while True:
                # Get output from the neural network
                output = model(state)[0]

                action = torch.zeros([number_of_actions], dtype=torch.float32).to(device)

                # Get action
                action_index = torch.argmax(output)
                action_index = action_index.to(device)
                
                action[action_index] = 1

                # Get next state
                image_data_1, reward, terminal = game_state.frame_step(action)
                image_data_1 = Image.resize_and_bgr2gray(image_data_1)
                image_data_1 = Image.image_to_tensor(image_data_1)
                image_data_1 = image_data_1.to(device)
                state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

                state = state_1
        finally:
            recorder.stop_rec()	# stop recording
            recorder.save_recording(f"./videos/{test_id}.avi") # saves the last recording
            cleanup()
            pygame.quit()
        

if __name__ == "__main__":
    
    test(test_id=0)
