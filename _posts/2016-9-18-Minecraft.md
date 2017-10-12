---
layout: post
title: Lets Play Minecraft&#58; Deep Reinforcement Learning
---
TLDR: I trained an AI in Minecraft, transferred it to a real robot, and ended up with a simple autonomous vehicle.
## Project Malmo
The Malmo platform is a sophisticated AI experimentation platform built on top of Minecraft, and designed to support fundamental research in artificial intelligence. This is super cool. Minecraft is an intuitive tool for quickly modeling an environment. We will be modeling a simple line for an agent to follow.


### The Environment


To generate a flat world use a [world generator](http://chunkbase.com/apps/superflat-generator#3;7,2*3,2;1;village) to make a world with white snow blocks as the ground. I manually added a black line on the ground with a couple right and left turns. Since the images will be converted to grayscale during processing it is best to use black and white.

<img src="/assets/minecraft/line.png" alt="Drawing" style="width: 300px; display: block; margin: 0 auto;"/>

<center><i>The line</i></center>

```xml
missionXML = '''<?xml version="1.0" encoding="UTF-8" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">

  <About>
    <Summary>Follow the line!</Summary>
  </About>
  
 <ServerSection>
    <ServerInitialConditions>
        <Time>
            <StartTime>6000</StartTime>
            <AllowPassageOfTime>false</AllowPassageOfTime>
        </Time>
        <Weather>clear</Weather>
        <AllowSpawning>false</AllowSpawning>
    </ServerInitialConditions>
    <ServerHandlers>
        <FileWorldGenerator src="C:\\Users\\You\\path\\to\\the\\Minecraft\\save\\file" />
        <ServerQuitFromTimeUp timeLimitMs="10000"/>
        <ServerQuitWhenAnyAgentFinishes />
    </ServerHandlers>
</ServerSection>
<AgentSection>
    <Name>Jason Bourne</Name>
    <AgentStart>
        <Placement x="0.5" y="5" z="0.5" yaw="90" pitch="30"/>
    </AgentStart>
    <AgentHandlers>
        <VideoProducer want_depth="false">
            <Width>''' + str(video_width) + '''</Width>
            <Height>''' + str(video_height) + '''</Height>
        </VideoProducer>
        <ContinuousMovementCommands turnSpeedDegs="720" />
        <RewardForTouchingBlockType>
            <Block reward="-100.0" type="snow" behaviour="oncePerBlock"/>
            <Block reward="1000.0" type="stained_hardened_clay" behaviour="oncePerBlock"/>
            <Block reward="1000.0" type="glowstone" behaviour="oncePerBlock"/>
        </RewardForTouchingBlockType>
    </AgentHandlers>
</AgentSection>
</Mission>'''
```
This is the XML for the mission. Most of the code should be self explanatory. We established a reward of -100 for touching snow and a reward of +1000 for standing on the black clay. To see exactly how Project Malmo is used check out the full code on Github. It should save you reading through a lot of documentation.


## Policy Gradients

Currently the most popular reinforcement learning algorithm is Q-learning(including [Deep Q-learning](https://www.nervanasys.com/demystifying-deep-reinforcement-learning/) of course). We can't use Q-learning (easily) in this example because the action space is continuous. In other words, I want to output a number from -1 to +1 which is how much the agent should turn. -1 is left, +1 is right, and 0 is forward. Q-learning only allows for a discrete action space: you can only output a couple of set values. I could have used Q-learning in this project if I only allowed the agent to choose left, right, or forward, but I wanted to experiment with a continuous action space where the agent can move any way it wants. 

I'm going to assume that you have a basic knowledge of [Neural Networks](https://www.youtube.com/playlist?list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU). In this project I used a watered down version of the Policy Gradient algorithm for a simple implementation. In the future I will be experimenting with more complex algorithms.

Policy Gradients are very direct. Unlike in Q-learning where the network outputs the value of the state, a Policy network outputs the action to be taken. This leads to an intuitive structure of the neural network. In our case it takes an image as input and outputs a number between -1 and +1 which is the amount the agent should turn. The basic loop for the algorithm is below:
 
<img src="/assets/minecraft/mdp.jpg" alt="Drawing" style="width: 300px; display: block; margin: 0 auto;"/>

<center><i>Markov Decision Process</i></center>

```
repeat
	get image s from environment
	proccess s
	get action a from the neural network given s
	execute action a in the environment
	receive reward r
until done	
```

The Markov decision process is the base of all reinforcement learning algorithms. The unique thing about policy gradients is how we train the neural network. 

### Network Architecture

[Convolution neural networks](http://cs231n.github.io/convolutional-networks/)(CNN) have been shown to perform very well when dealing with images. We will use [Keras](https://keras.io/) to quickly define a standard CNN. 

```python
createModel('relu', 0.001)
def createModel(self, activationType, learningRate):
            model = Sequential()
            model.add(Convolution2D(8, 3, 3, input_shape=(1, video_width, video_height)))
            model.add(Activation(activationType))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Convolution2D(4, 3, 3))
            model.add(Activation(activationType))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(1, init='lecun_uniform'))
            model.add(Activation("linear"))
            optimizer = optimizers.RMSprop(lr=learningRate, rho=0.9, epsilon=1e-06)
            model.compile(loss="mse", optimizer=optimizer)
            return model
```

### Training the Network

After every episode (runs for 10 seconds before resetting) is over, we sum the rewards for each timestep. We want to make the total rewards as high as possible because that means the agent is following the line. We also remember all of the previous episodes total rewards. By comparing the current total rewards to the historical average we can determine if the network is improving or not. A simple measure of how good or bad the network has done is the following:
```
Advantage = Total rewards of this episode - Average of all total rewards
```
This yields a positive result if good, and negative when bad. Now to update the network with this knowledge.

To do this we remember all of the states and corresponding actions taken during the episode and store them. Lets assume we calculate the advantage of this episode to be positive. This means the actions we took in this episode are on average better than the actions in previous episodes. So we want to encourage taking the actions in this episode. We do this by training the neural network like this:
```
initialize empty lists X and Y
for every pair (state, action)
	y = action * advantage
	add state to list X
	add y to list Y
end
train model on batch X and Y (*)
```
For every state we want to encourage the action it took so we train it to perform the same action multiplied by the advantage. Multiplying by the advantage makes the neural net train faster if advantage is high and slower if low. And that's it! If the advantage is negative, meaning the actions were below average, action * advantage trains the network to do the opposite of the action since it is negated. Note that this algorithm only works when the actions are opposite to each other (left/right). In more complex problems it will be necessary to use a more standard policy gradient algorithm.


## Putting it All Together

There are some details which I have left out. Here are the interesting bits.

### Processing the images
include example of image before and after processing
```python
def processFrame(frame):
    frame = np.reshape(frame, (video_height, video_width, 3))
    img = Image.fromarray(frame).convert('L')
    f = np.array(img, dtype=np.uint8)
    return f
```
The frame returned by Malmo is formatted strangely and needs to be processed before feeding it to the network. Here we use numpy to reshape the array into a more standard format. We then use the Python imaging library (PIL) to convert this array into an image. PIL then converts this image into grayscale. Finally numpy converts the image back into an array which is now ready to be used in the neural network.

### Syntax

```python
def step(action):
    agent_host.sendCommand( "turn " + str(action) )
    world_state = agent_host.getWorldState()
    ss = np.zeros(shape=(video_height, video_width))
    while world_state.number_of_video_frames_since_last_state < 1 and world_state.is_mission_running:
        #logger.info("Waiting for frames...")
        time.sleep(0.05)
        world_state = agent_host.getWorldState()
    #logger.info("Got frame!")
    if len(world_state.video_frames) > 0:
        ss = processFrame(world_state.video_frames[0].pixels)
    r = sum(r.getValue() for r in world_state.rewards)
    done = world_state.is_mission_running
    return ss, r, done
```
This is the code which is run every timestep. A lot of the syntax was difficult to find in the documentation and the examples so I hope this helps a bit. 


All of the code can be found on [Github](). 


## Results
The end results are pretty good. The agent cuts corners, but it is clear that it is indeed following the line consistently. It only takes about an hour to achieve decent results with a GPU. I am sure the agent will only get better if you are more patient than I am. 

## The real world
The agent worked well enough in Minecraft that I wanted to test it out it real life. I built a simple Lego robot and attached my phone to it. The phone takes pictures of the track and sends it to my computer. The computer processes the image and runs it through the model we trained in Minecraft. The result is then sent to the robot and the appropriate action is taken. Miraculously this actually worked without any modifications to the model. The robot is forced to move very slowly due to communication lag. The results are good enough for me though. I trained an AI in Minecraft, transferred it to a real robot, and ended up with a simple autonomous vehicle.

<img src="/assets/minecraft/robot.gif" alt="Drawing" style="width: 300px; display: block; margin: 0 auto;"/>

## Final Thoughts

The task we just solved is a very simple task. You don't really even need to use machine learning to follow a black line on a white background. In the future I might experiment with adding noise to the background or adding obstacles. In Minecraft you can do pretty much anything so there are plenty of things to try. 

Project Malmo is a super cool tool to use and it opens up a lot of possibilities for future research. Video games in general are now attracting the attention of researchers. There has been some [recent research](http://hackaday.com/2016/09/16/grand-theft-auto-v-used-to-teach-self-driving-ai/) in GTA V (a much more realistic game) to train self driving cars. I'm exited to see the applications and results of research done in video games.
