# Portfolio
This portfolio contains sample projects I've worked on. The file "torch_rl_fight" is the PyTorch-based reinforcement learning script I've used for a private project in which I've built an ancient battle simulator where two ancient armies fight against each other. At each time step, each fighter receives its own observation of the environment (numpy array with 8 values: one potential field value for each of the 6 fighter types, its own type and its team number) and can then move up,down,right,left or stay). The RL script uploaded uses this environment to find effective fighting strategies using self-play. Because I've spent a considerable amount of time on the project and plan to create YouTube-Videos about it, I've decided to upload only the RL script and keep the environment private.

The "Jax_Actor_Critic" file is a translation of the [TensorFlow Actor-Critic tutorial]([url](https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic)) to Jax. I've built this when Jax was new and there was little documentation/ example scripts of Jax online. We later used this script as baseline for the RL part of [SwarmRL]([url](https://github.com/SwarmRL/SwarmRL)), a Python package I'm a co-creator of and which can be used to do Physics research using reinforcement learning (RL) without having prior knowledge of RL.
