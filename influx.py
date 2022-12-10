import func
from sympy import *
from matplotlib import pyplot as plt
import random

x, y = symbols('x y')
f = Function('f')
init_printing(use_unicode=True)


def v_linha(v_zero, t_zero, unidades_temporais, h):
    """
    Com base nos valores físicos e nas condições iniciais, gera a EDO a ser calculada, feito isso chama a função
    calcula(), para atravez dos métodos numéricos, calcular a solução da EDO
    :param v_zero: volume inicial
    :param t_zero: tempo inicial
    :param unidades_temporais: intervalo de tempo a serem feitos os calculos
    :param h: passo entre o intervalo de tempo
    """
    tempo = 0
    Q1 = []
    Q2 = []

    for i in range(unidades_temporais + 1):
        if 0 <= tempo < 20:
            q1 = 110
        elif 20 <= tempo < 40:
            q1 = 100
        elif 40 <= tempo < 80:
            q1 = 95
        else:
            q1 = 100
        Q1.append(q1)

        if 0 <= tempo < 30:
            q2 = 100
        elif 30 <= tempo < 40:
            q2 = 95
        elif 40 <= tempo < 70:
            q2 = 105
        else:
            q2 = 85
        Q2.append(q2)

        tempo += 1

    # vazamento
    v_nominal = 0
    vlim = 10
    t_atraso = 50
    tau = 0.05

    V_linha = []
    Vazamentos = []
    Ruido = []
    for i in range(unidades_temporais + 1):

        if t_atraso - x.subs(x, i) >= 0:
            theta = 0
        else:
            theta = t_atraso - x

        vazamento = vlim - (vlim - v_nominal) * exp(tau * theta)
        Vazamentos.append(vazamento.subs(x, i))

        # v_linha
        ruido = round(random.uniform(-3, 3), 15)
        Ruido.append(ruido)

        v_linha = Q1[i] - Q2[i] - vazamento + ruido

        V_linha.append(v_linha.subs(x, i))

    calcula(v_zero, t_zero, h, unidades_temporais, V_linha, Q1, Q2, Vazamentos, Ruido)


def calcula(y_zero, x_zero, h, num_repet, v_linha, q1, q2, vazamentos, ruidos):
    """
    Utiliza o PVI a ser calculado, utilizando os métodos numéricos para solução de EDOs, depois envia as soluções
    chamando a função graficos_influx() para criar os gráficos.
    :param y_zero: volume inicial
    :param x_zero: tempo inicial
    :param h: passo entre o intervalo de tempo
    :param num_repet: intervalo de tempo a serem feitos os calculos
    :param v_linha: EDOs
    :param q1: Valores de Q1
    :param q2: Valores de Q2
    :param vazamentos: Vazamentos
    :param ruidos: Ruidos
    """
    valores_t = []
    for i in range(len(v_linha)):
        valores_t.append(i)

    valores = [valores_t]

    valores.append(func.euler(y_zero, x_zero, h, num_repet, v_linha))
    valores.append(func.euler_mel(y_zero, x_zero, h, num_repet, v_linha))
    valores.append(func.euler_mod(y_zero, x_zero, h, num_repet, v_linha))
    valores.append(func.gen_seg_ord_alfa(y_zero, x_zero, h, num_repet, v_linha, 1 / 3))
    valores.append(func.gen_seg_ord_alfa(y_zero, x_zero, h, num_repet, v_linha, 1 / 4))
    valores.append(func.dormand_price_fixo(y_zero, x_zero, h, num_repet, v_linha))

    graficos_influx(valores, y_zero, q1, q2, vazamentos, ruidos, v_linha)


def graficos_influx(valores, v_zero, q1, q2, vazamentos, ruidos, v_linha):
    """
    Cria os gráficos com base nos resultados da EDOs
    :param valores: conjunto de y solução das EDOs
    :param v_zero: volume inicial
    :param q1: Q1
    :param q2: Q2
    :param vazamentos: Vazamentos
    :param ruidos: Ruidos
    :param v_linha: EDOs
    """
    plt.style.use('seaborn')

    comparacao, compa = plt.subplots()

    euler_menos = []
    euler_mel_menos = []
    euler_mod_menos = []
    gen_seg_ord_terco_menos = []
    gen_seg_ord_quarto_menos = []
    dormand_price_menos = []
    for i in range(len(valores[0])):
        euler_menos.append(valores[1][i] - v_zero)
        euler_mel_menos.append(valores[2][i] - v_zero)
        euler_mod_menos.append(valores[3][i] - v_zero)
        gen_seg_ord_terco_menos.append(valores[4][i] - v_zero)
        gen_seg_ord_quarto_menos.append(valores[5][i] - v_zero)
        dormand_price_menos.append(valores[6][i] - v_zero)

    valores_menos_v_zero = [euler_menos, euler_mel_menos, euler_mod_menos, gen_seg_ord_terco_menos,
                            gen_seg_ord_quarto_menos, dormand_price_menos]

    compa.plot(valores[0], valores_menos_v_zero[0], label='Euler', linewidth='3', color='#1A5274',
               alpha=0.9)
    compa.plot(valores[0], valores_menos_v_zero[1], label='Euler melhorado', linewidth='3',
               color='#B892FF')
    compa.plot(valores[0], valores_menos_v_zero[2], label='Euler modficado', linewidth='3',
               color='#FEC620')
    compa.plot(valores[0], valores_menos_v_zero[3], label='Genérico de segunda ordem com alfa = 1/3', linewidth='3',
               color='#BAFF29')
    compa.plot(valores[0], valores_menos_v_zero[4], label='Genérico de segunda ordem com alfa = 1/4', linewidth='3',
               color='#2AB7CA', linestyle='--')
    compa.plot(valores[0], valores_menos_v_zero[5], label='Dormand Price com passo fixo', linewidth='3',
               color='#FE4A49', linestyle=':')

    compa.legend(fontsize='medium')
    compa.set_title(f"Métodos numéricos para subtração do fluxo pelo volume inicial(Q - Q_zero)")

    plt.tight_layout()
    plt.savefig(f'Comparação métodos numéricos para o fluxo')

    graf, flux_vol = plt.subplots()

    flux_vol.plot(valores[0], valores_menos_v_zero[0], label='V - V_zero (Euler)', color='#BAFF29')
    flux_vol.plot(valores[0], q1, label='Afluente', color='#2AB7CA')
    flux_vol.plot(valores[0], q2, label='Efluente', color='#FEC620')
    flux_vol.plot(valores[0], v_linha, label='Q', color='#1A5274')
    flux_vol.plot(valores[0], vazamentos, label='Vazamentos', color='#B892FF')
    flux_vol.plot(valores[0], ruidos, label='Ruídos', color='#FE4A49')

    flux_vol.legend(fontsize='medium')
    flux_vol.set_title(f"Fluxos e volume do reservátorio")

    plt.tight_layout()
    plt.savefig(f'Fluxos e volume do reservátorio')
