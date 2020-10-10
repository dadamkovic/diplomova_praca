# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Pomocná funkcia na vizualizáciu výsledkov.
def draw_results(title):
    if not hasattr(draw_results, "counter"):
        draw_results.counter = 1
    else:
        draw_results.counter += 1
    
    # Pozrieme sa aké výstupy dáva neurón na začiatku - ešte pred učením.
    plt.figure(); plt.grid()
    plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
    
    plt.xlabel("vstup $x_1$")
    plt.ylabel("vstup $x_2$")
    
    print(title)
    
    # Pre každý prvok dátovej množiny.
    for x, d in zip(X, D):
        u = np.dot(x, W)
        y = sign(u)
        
        plt.scatter(x[0], x[1], s=100, c = 'k' if d else 'w')
        print("y = {} \t d = {}".format(y, d))
        
    # Priamka, ktorú neurón realizuje.
    plt.plot([(0.5 * W[1] - W[2]) / W[0], -0.5],
             [(-1.5 * W[1] - W[2]) / W[0], 1.5], 'r')
    
    plt.tight_layout()
    plt.gcf().set_size_inches(4, 3)
    plt.savefig('delta_rule_example_' + str(draw_results.counter) + '.pdf',
        dpi=400, bbox_inches='tight', pad_inches=0)    
    
    plt.title(title)
    print('')

# Znamienková aktivačná funkcia.
def sign(x):
    if x > 0: return 1
    else: return 0

# Náhrada derivácie znamienkovej funkcie (nie je diferencovateľná).
def sign_derivative(x):
    return 1

# Množina vstupov. Posledný vstup je bias - je vždy jeden. Jeho váhou
# realizujeme prahový potenciál.
X = [
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]

# Požadované výstupy - funkcia OR.
D = [
    0,
    1,
    1,
    1
]

# Zvolíme nejaké počiatočné váhy.
W = [0.1, 0.75, -0.5]

# Rýchlosť učenia.
learning_rate = 0.1
# Počet krokov učenia.
num_steps = 1000

draw_results("Pred učením")

# Učenie pomocou delta pravidla.
for i in range(num_steps):
    for x, d in zip(X, D):
        u = np.dot(x, W)
        y = sign(u)
        
        delta = (d - y) * sign_derivative(u)
                
        for j in range(len(W)):
            dW = learning_rate * delta * x[j]
            W[j] += dW
        
draw_results("Po učení")