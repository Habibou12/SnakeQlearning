# IA Learning to solve Snake
Training an ia to play snake with Q-learning 
### Snake Game
This snake is display using pygame, they is two option that you can use when creating the game instance Game(True), will enabled the display of the game, usefull to see what the ia can to an Game(False) to train the ia model.
The grid is 40*40 so they is 1600 tiles
### Test the game
You can run the main.py script to train the ia from scratch, the test.py model is for testing the ia that I have train and finally you can run the Game.py to play yourself this basic snake game.
### The IA Model
The IA Model use 3 layer fully connect architecture : layer1(11,200) layer2(200,200) layer3 (200,3)
layer1 and 2 use the relu activation fonction, and with AdamW
the 3 output are move left, right and forward.
 The training method is Q learning.

Bellman equation : Qtarget(s, a) = R + gamma x Qmax(s', a'),

The model is train off-policy, because he will try to learn the optimal Q policy.
Q(s,a) = Q(s,a) + lr x (Qtarget(s,a) - Q(s,a))

The IA is giving as a state : ForwardCollision, LeftCollision, RightCollision, is_dir_l,is_dir_r, is_dir_u, is_dir_d, is_apple_r, is_apple_l, is_apple_d, is_apple_u. 
 The probleme with this is that the ia model can't know where is body is and will lock himself in.
 This ia model is simple and primitve but work, maybe not perfectly but it work and thats what matter



