import gymnasium as gym
import numpy as np
import pygame

ENABLE_WIND = True
WIND_POWER = 20.0
TURBULENCE_POWER = 0.0
GRAVITY = -10.0
RENDER_MODE = 'human'
RENDER_MODE = None #selecione esta opção para não visualizar o ambiente (testes mais rápidos)
EPISODES = 1000

env = gym.make("LunarLander-v3", render_mode=RENDER_MODE,
    continuous=True, gravity=GRAVITY, 
    enable_wind=ENABLE_WIND, wind_power=WIND_POWER, 
    turbulence_power=TURBULENCE_POWER)


def check_successful_landing(observation):
    x = observation[0]
    vy = observation[3]
    theta = observation[4]
    contact_left = observation[6]
    contact_right = observation[7]

    legs_touching = contact_left == 1 and contact_right == 1

    on_landing_pad = abs(x) <= 0.2

    stable_velocity = vy > -0.2
    stable_orientation = abs(theta) < np.deg2rad(20)
    stable = stable_velocity and stable_orientation
 
    if legs_touching and on_landing_pad and stable:
        print("Aterragem bem sucedida!")
        return True

    print("Aterragem falhada!")        
    return False
        
def simulate(steps=1000,seed=None, policy = None):    
    observ, _ = env.reset(seed=seed)
    for step in range(steps):
        action = policy(observ)

        observ, _, term, trunc, _ = env.step(action)

        if term or trunc:
            break

    success = check_successful_landing(observ)
    return step, success

# --- THRESHOLDS ---

TARGET_ANGLE   = 0.3 # ângulo alvo da nave

# --- PERCEPTIONS ---

def _sigma(obs):

    x_pred = obs[0] + obs[2]

    if x_pred > 0:
        angle = TARGET_ANGLE
    elif x_pred < 0:
        angle = -TARGET_ANGLE
    else:
        angle = 0.0

    return (obs[4] - angle) + obs[5]

def in_contact(obs):
    return obs[6] == 1 or obs[7] == 1

def tilt_positive(obs):
    return _sigma(obs) > 0

def tilt_negative(obs):
    return _sigma(obs) < 0

def altitude_low(obs):
    return obs[1] < abs(obs[0])

def falling_fast(obs):
    return obs[3] < -0.20

# --- ACTIONS ---

def main_on(): return 1.0
def main_off(): return 0   

def left_on(): return 1.0
def lateral_off(): return 0.0 
def right_on(): return -1.0

# --- AGENT ---

def reactive_agent(observation):

    # P1: aterragem
    if in_contact(observation):                                                                   # P1
        return [main_off(), lateral_off()]                                                        

    # P2-P3: inclinação com altitude crítica ou descida rápida
    if tilt_positive(observation) and (altitude_low(observation) or falling_fast(observation)):   # P2
        return [main_on(), left_on()]
    if tilt_negative(observation) and (altitude_low(observation) or falling_fast(observation)):   # P3
        return [main_on(), right_on()]

    # P4-P5: inclinação com altitude segura
    if tilt_positive(observation):                                                                # P4
        return [main_off(), left_on()]
    if tilt_negative(observation):                                                                # P5                                                
        return [main_off(), right_on()]

    # P6-P7: orientado
    if altitude_low(observation) or falling_fast(observation):                                    # P6
        return [main_on(), lateral_off()]
                                        
    return [main_off(), lateral_off()]                                                            # P7
    
def keyboard_agent(observation):
    action = [0,0] 
    keys = pygame.key.get_pressed()
    
    print('observação:',observation)

    if keys[pygame.K_UP]:  
        action =+ np.array([1,0])
    if keys[pygame.K_LEFT]:  
        action =+ np.array( [0,-1])
    if keys[pygame.K_RIGHT]: 
        action =+ np.array([0,1])

    return action
    
success = 0.0
steps = 0.0
for i in range(EPISODES):
    st, su = simulate(steps=1000000, policy=reactive_agent)

    if su:
        steps += st
    success += su
    
    if su>0:
        print('Média de passos das aterragens bem sucedidas:', steps/success*100)
    print('Taxa de sucesso:', success/(i+1)*100)
    
