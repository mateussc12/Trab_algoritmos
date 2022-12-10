"""
Código do grupo: Mateus Souza Coelho, Felipe Antonio Moreira Silva e Kaique de Oliveira Barcelos.
Para executar o código são necessárias as seguintes bibliotecas : matplotlib, sympy, scipy e random.
Caso alguma delas não esteja instalada, ir no terminal e escrever:
pip install *nome da biblioteca*
"""
import func
import influx
from sympy import *

x, y = symbols('x y')
f = Function('f')
init_printing(use_unicode=True)


#   Primeira parte do trabalho usa a biblioteca "func"

# Primeiro PVI
y_zero = 1
x_zero = 0
h = 0.1
num_repet = 10
y_linha = -f(x)

func.calcula_main(y_zero, x_zero, h, num_repet, y_linha, 1)

# Segundo PVI
y_zero = 4
x_zero = 2
h = 0.1
num_repet = 10
y_linha = (x + f(x) + 1) / (2 * x)

func.calcula_main(y_zero, x_zero, h, num_repet, y_linha, 2)

# Terceiro PVI
y_zero = 1
x_zero = 0
h = 0.1
num_repet = 10
y_linha = f(x) * (pow(x, 2) - 1)

func.calcula_main(y_zero, x_zero, h, num_repet, y_linha, 3)


#   Segunda parte do trabalho usa a biblioteca "influx"

# Problema prático
v_zero = 500
t_zero = 0
unidades_temporais = 100
h = 1

influx.v_linha(v_zero, t_zero, unidades_temporais, h)
