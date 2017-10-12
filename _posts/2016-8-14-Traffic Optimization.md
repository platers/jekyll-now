---
layout: post
title: Traffic Optimization
---
One day while stuck in traffic I wondered if I could solve this problem. This is my attempt.

I used the SUMO open source software to simulate a simple road network consisting of a 2x2 grid of traffic lights. I then gathered data on multiple simple algorithms to control the lights. I then implemented a reinforcement learning algorithm to control the lights. I trained it to optimize for waiting times and CO2 emissions. After many days of fiddling with parameters I ended up with a 5% improvement in both CO2 emissions and waiting times. This is a somewhat significant result, but probably not worth the hassle of replacing the existing algorithms in the real world. I learned a lot about practical machine learning.

Here is a video of training in progress.
<img src="/assets/traffic/demo.gif" alt="Drawing" style="width: 600px; display: block; margin: 0 auto;"/>
