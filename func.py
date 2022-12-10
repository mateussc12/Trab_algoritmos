from sympy import *
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp


x, y = symbols('x y')
f = Function('f')
init_printing(use_unicode=True)


def euler(y_zero, x_zero, h, num_repet, y_linha):
    """
    Calula a EDO utilizando o método de Euler
    euler => y(n+1) = yn + hf(xn, yn)
    :param y_zero: y inicial
    :param x_zero: x inicial
    :param h: passo
    :param num_repet: número de répetições do passo
    :param y_linha: EDO
    :return: valores de y do Euler
    """
    y_valores = [y_zero]
    for i in range(num_repet):
        x_n = x_zero + i * h
        y_n = y_valores[i]

        try:
            calc_euler = y_n + h * y_linha.subs([(x, x_n), (y, y_n)])

        except AttributeError:
            calc_euler = y_n + h * y_linha[i].subs([(x, x_n), (y, y_n)])

        y_valores.append(round(calc_euler, 15))

    return y_valores


def euler_mel(y_zero, x_zero, h, num_repet, y_linha):
    """
    Calula a EDO utilizando o método de Euler melhorado
    euler_mel => y(n+1) = yn + (h/2) [f(xn, yn) + f(xn + h, yn + hf(xn, yn))]
    :param y_zero: y inicial
    :param x_zero: x inicial
    :param h: passo
    :param num_repet: número de répetições do passo
    :param y_linha: EDO
    :return: valores de y do Euler melhorado
    """
    y_valores = [y_zero]
    for i in range(num_repet):
        x_n = x_zero + i * h
        y_n = y_valores[i]

        try:
            calc_euler_mel = y_n + (h / 2) * (y_linha.subs([(x, x_n), (y, y_n)]) +
                                              y_linha.subs([(x, x_n + h),
                                                            (y, y_n + h * y_linha.subs([(x, x_n), (y, y_n)]))]))
        except AttributeError:
            calc_euler_mel = y_n + (h / 2) * (y_linha[i].subs([(x, x_n), (y, y_n)]) +
                                              y_linha[i].subs([(x, x_n + h),
                                                               (y, y_n + h * y_linha[i].subs([(x, x_n), (y, y_n)]))]))

        y_valores.append(round(calc_euler_mel, 15))

    return y_valores


def euler_mod(y_zero, x_zero, h, num_repet, y_linha):
    """
    Calula a EDO utilizando o método de Euler modificado
    euler_mod => y(n+1) = yn + hf(xn + h/2, yn + h/2 * f(xn, yn))
    :param y_zero: y inicial
    :param x_zero: x inicial
    :param h: passo
    :param num_repet: número de répetições do passo
    :param y_linha: EDO
    :return: valores de y do Euler modificado
    """
    y_valores = [y_zero]
    for i in range(num_repet):
        x_n = x_zero + i * h
        y_n = y_valores[i]

        try:
            calc_euler_mod = y_n + h * y_linha.subs([(x, x_n + (h / 2)),
                                                     (y, y_n + ((h / 2) * y_linha.subs([(x, x_n), (y, y_n)])))])
        except AttributeError:
            calc_euler_mod = y_n + h * y_linha[i].subs([(x, x_n + (h / 2)),
                                                        (y, y_n + ((h / 2) * y_linha[i].subs([(x, x_n), (y, y_n)])))])

        y_valores.append(round(calc_euler_mod, 15))

    return y_valores


def gen_seg_ord_alfa(y_zero, x_zero, h, num_repet, y_linha, alpha):
    """
    Calula a EDO utilizando o método genérico de segunda ordem com alfa 1/3 e 1/4
    :param y_zero: y inicial
    :param x_zero: x inicial
    :param h: passo
    :param num_repet: número de répetições do passo
    :param y_linha: EDO
    :param alpha: alpha utlizado
    :return: Valores de y do genérico de segunda ordem
    """
    c = [0, alpha]
    b = [1 - 1 / (2 * alpha), 1 / (2 * alpha)]

    y_valores = [y_zero]
    for i in range(num_repet):
        x_n = x_zero + i * h
        y_n = y_valores[i]

        try:
            rk_2 = y_n + h * (b[0] * y_linha.subs([(x, x_n), (y, y_n)]) + b[1] * y_linha.subs(
                [(x, x_n + c[1] * h), (y, y_n + h * alpha * y_linha.subs([(x, x_n), (y, y_n)]))]))
        except AttributeError:
            rk_2 = y_n + h * (b[0] * y_linha[i].subs([(x, x_n), (y, y_n)]) + b[1] * y_linha[i].subs(
                [(x, x_n + c[1] * h), (y, y_n + h * alpha * y_linha[i].subs([(x, x_n), (y, y_n)]))]))

        y_valores.append(round(rk_2, 15))

    return y_valores


def dormand_price_fixo(y_zero, x_zero, h, num_repet, y_linha):
    """
    Calula a EDO utilizando o método de Dormand=Price
    :param y_zero: y inicial
    :param x_zero: x inicial
    :param h: passo
    :param num_repet: número de répetições do passo
    :param y_linha: EDO
    :return: valores de y do de dormand_price fixo
    """
    matriz = [[0, 0, 0, 0, 0, 0, 0], [1 / 5, 0, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
              [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0], [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
              [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
              [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]]

    c = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]
    b2 = matriz[6]

    y_valores2 = [y_zero]

    try:
        for i in range(num_repet):
            x_n = x_zero + i * h
            y_n2 = y_valores2[i]

            k1 = y_linha.subs([(x, x_n + c[0] * h), (y, y_n2)])
            k2 = y_linha.subs([(x, x_n + c[1] * h), (y, y_n2 + h * matriz[1][0] * k1)])
            k3 = y_linha.subs([(x, x_n + c[2] * h), (y, y_n2 + h * (matriz[2][0] * k1 + matriz[2][1] * k2))])
            k4 = y_linha.subs(
                [(x, x_n + c[3] * h), (y, y_n2 + h * (matriz[3][0] * k1 + matriz[3][1] * k2 + matriz[3][2] * k3))])
            k5 = y_linha.subs(
                [(x, x_n + c[4] * h),
                 (y, y_n2 + h * (matriz[4][0] * k1 + matriz[4][1] * k2 + matriz[4][2] * k3 + matriz[4][3] * k4))])
            k6 = y_linha.subs([(x, x_n + c[5] * h),
                               (y,
                                y_n2 + h * (matriz[5][0] * k1 + matriz[5][1] * k2 + matriz[5][2] * k3 + matriz[5][
                                    3] * k4 +
                                            matriz[5][4] * k5))])
            k7 = y_linha.subs([(x, x_n + c[6] * h), (
                y,
                y_n2 + h * (matriz[6][0] * k1 + matriz[6][1] * k2 + matriz[6][2] * k3 + matriz[6][3] * k4 + matriz[6][
                    4] * k5 + matriz[6][5] * k6))])

            dp2 = y_n2 + h * (b2[0] * k1 + b2[1] * k2 + b2[2] * k3 + b2[3] * k4 + b2[4] * k5 + b2[5] * k6 + b2[6] * k7)

            y_valores2.append(round(dp2, 15))

        b = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]

        y_valores = [y_zero]

        for i in range(num_repet):
            x_n = x_zero + i * h
            y_n = y_valores[i]

            k1 = y_linha.subs([(x, x_n + c[0] * h), (y, y_n)])
            k2 = y_linha.subs([(x, x_n + c[1] * h), (y, y_n + h * matriz[1][0] * k1)])
            k3 = y_linha.subs([(x, x_n + c[2] * h), (y, y_n + h * (matriz[2][0] * k1 + matriz[2][1] * k2))])
            k4 = y_linha.subs(
                [(x, x_n + c[3] * h), (y, y_n + h * (matriz[3][0] * k1 + matriz[3][1] * k2 + matriz[3][2] * k3))])
            k5 = y_linha.subs(
                [(x, x_n + c[4] * h),
                 (y, y_n + h * (matriz[4][0] * k1 + matriz[4][1] * k2 + matriz[4][2] * k3 + matriz[4][3] * k4))])
            k6 = y_linha.subs([(x, x_n + c[5] * h),
                               (
                                   y, y_n + h * (matriz[5][0] * k1 + matriz[5][1] * k2 + matriz[5][2] * k3 + matriz[5][
                                       3] * k4 +
                                                 matriz[5][4] * k5))])
            k7 = y_linha.subs([(x, x_n + c[6] * h), (
                y, y_n + h * (matriz[6][0] * k1 + matriz[6][1] * k2 + matriz[6][2] * k3 + matriz[6][3] * k4 + matriz[6][
                    4] * k5 + matriz[6][5] * k6))])

            dp = y_valores2[i] + h * (b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6 + b[6] * k7)

            y_valores.append(round(dp, 15))
    except AttributeError:
        for i in range(num_repet):
            x_n = x_zero + i * h
            y_n2 = y_valores2[i]

            k1 = y_linha[i].subs([(x, x_n + c[0] * h), (y, y_n2)])
            k2 = y_linha[i].subs([(x, x_n + c[1] * h), (y, y_n2 + h * matriz[1][0] * k1)])
            k3 = y_linha[i].subs([(x, x_n + c[2] * h), (y, y_n2 + h * (matriz[2][0] * k1 + matriz[2][1] * k2))])
            k4 = y_linha[i].subs(
                [(x, x_n + c[3] * h), (y, y_n2 + h * (matriz[3][0] * k1 + matriz[3][1] * k2 + matriz[3][2] * k3))])
            k5 = y_linha[i].subs(
                [(x, x_n + c[4] * h),
                 (y, y_n2 + h * (matriz[4][0] * k1 + matriz[4][1] * k2 + matriz[4][2] * k3 + matriz[4][3] * k4))])
            k6 = y_linha[i].subs([(x, x_n + c[5] * h),
                                  (y,
                                   y_n2 + h * (matriz[5][0] * k1 + matriz[5][1] * k2 + matriz[5][2] * k3 + matriz[5][
                                       3] * k4 +
                                               matriz[5][4] * k5))])
            k7 = y_linha[i].subs([(x, x_n + c[6] * h), (
                y,
                y_n2 + h * (matriz[6][0] * k1 + matriz[6][1] * k2 + matriz[6][2] * k3 + matriz[6][3] * k4 + matriz[6][
                    4] * k5 + matriz[6][5] * k6))])

            dp2 = y_n2 + h * (b2[0] * k1 + b2[1] * k2 + b2[2] * k3 + b2[3] * k4 + b2[4] * k5 + b2[5] * k6 + b2[6] * k7)

            y_valores2.append(round(dp2, 15))

        b = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]

        y_valores = [y_zero]

        for i in range(num_repet):
            x_n = x_zero + i * h
            y_n = y_valores[i]

            k1 = y_linha[i].subs([(x, x_n + c[0] * h), (y, y_n)])
            k2 = y_linha[i].subs([(x, x_n + c[1] * h), (y, y_n + h * matriz[1][0] * k1)])
            k3 = y_linha[i].subs([(x, x_n + c[2] * h), (y, y_n + h * (matriz[2][0] * k1 + matriz[2][1] * k2))])
            k4 = y_linha[i].subs(
                [(x, x_n + c[3] * h), (y, y_n + h * (matriz[3][0] * k1 + matriz[3][1] * k2 + matriz[3][2] * k3))])
            k5 = y_linha[i].subs(
                [(x, x_n + c[4] * h),
                 (y, y_n + h * (matriz[4][0] * k1 + matriz[4][1] * k2 + matriz[4][2] * k3 + matriz[4][3] * k4))])
            k6 = y_linha[i].subs([(x, x_n + c[5] * h),
                                  (
                                      y,
                                      y_n + h * (matriz[5][0] * k1 + matriz[5][1] * k2 + matriz[5][2] * k3 + matriz[5][
                                          3] * k4 +
                                                 matriz[5][4] * k5))])
            k7 = y_linha[i].subs([(x, x_n + c[6] * h), (
                y, y_n + h * (matriz[6][0] * k1 + matriz[6][1] * k2 + matriz[6][2] * k3 + matriz[6][3] * k4 + matriz[6][
                    4] * k5 + matriz[6][5] * k6))])

            dp = y_valores2[i] + h * (b[0] * k1 + b[1] * k2 + b[2] * k3 + b[3] * k4 + b[4] * k5 + b[5] * k6 + b[6] * k7)

            y_valores.append(round(dp, 15))

    return y_valores


def dormand_price_adap(y_zero, x_zero, h, num_repet, contador):
    """
    Calcula a EDO usando Dormand-Price com passos adaptativos
    :param y_zero: y inicial
    :param x_zero: x inicial
    :param h: passo
    :param num_repet: número de répetições do passo
    :param contador: contador
    :return: valores de y e x de dormand_price adaptativo
    """

    if contador == 1:
        def dy_dt(t, y):
            return -y
    elif contador == 2:
        def dy_dt(t, y):
            return (t + y + 1) / (2 * t)
    else:
        def dy_dt(t, y):
            return y * (pow(t, 2) - 1)

    sol = solve_ivp(dy_dt, [x_zero, x_zero + h * num_repet], [y_zero], first_step=h)

    sol_t = list(sol.t)
    sol_y = list(sol.y[0])

    for i in range(len(sol_t)):
        sol_t[i] = round(sol_t[i], 15)
        sol_y[i] = round(sol_y[i], 15)

    valores = [sol_t, sol_y]

    return valores


def sol_real(y_linha, y_zero, x_zero):
    """
    Resolve uma EDO com condição inicial
    :param y_linha: função(dy/dx = função)
    :param y_zero: y inicial
    :param x_zero: x inicial
    :return: solução da EDO, sendo uma equação simbolica do sympy(Eq())
    """
    ode = Eq(Derivative(f(x), x), y_linha)
    sol = dsolve(ode, f(x), ics={f(x_zero): y_zero})
    return sol


def converte_eq_em_naosimbolico(sol_real_sympy, conjunto_x, y_string=False):
    """
    Converte Uma equação simbolica do sympy(Eq()) em um float python
    :param sol_real_sympy: Equação sympy
    :param conjunto_x: Conjunto de x a serem substituidos na função y(x)
    :param y_string: String com a função y(usado para o título dos gráficos e tabelas)
    :return: caso y_string=False, retorna os valores de y, se não retorna y_string que é uma string da y' usada para
    fins de título nos gráficos e tabelas.
    """
    aux = str(sol_real_sympy)
    sol_string = aux[9:len(aux) - 1]

    sol = sympify(sol_string)

    conjunto_y = []
    for i in range(len(conjunto_x)):
        conjunto_y.append(round(float(sol.subs(x, conjunto_x[i])), 15))

    if y_string is False:
        return conjunto_y
    else:
        return sol


def calcula_main(y_zero, x_zero, h, num_repet, y_linha, contador):
    """
    Utiliza o PVI a ser calculado, utilizando os métodos numéricos para solução de EDOs, depois envia as soluções
    chamando as funções grafico() e print_dados() para respectivamente criar os gráficos e tabelas.
    :param y_zero: y inicial
    :param x_zero: x inicial
    :param h: passo
    :param num_repet: número de répetições do passo
    :param contador: contador para ser salvo no gráfico
    :param y_linha: EDO
    """
    sol_real_sympy = sol_real(y_linha, y_zero, x_zero)

    aux = str(y_linha)
    y_novo = aux.replace("f(x)", "y")
    y_linha = sympify(y_novo)

    valores = []

    conjunto_x = []
    for i in range(num_repet + 1):
        conjunto_x.append(x_zero + i * h)

    titulo = converte_eq_em_naosimbolico(sol_real_sympy, conjunto_x, True)

    #   valores é uma lista de listas contendo [conjunto_x, Euler, Euler_mel, Euler_mod, y_real, 2ord a=1/3, 2ord a=1/4]
    valores.append(conjunto_x)
    valores.append(euler(y_zero, x_zero, h, num_repet, y_linha))
    valores.append(euler_mel(y_zero, x_zero, h, num_repet, y_linha))
    valores.append(euler_mod(y_zero, x_zero, h, num_repet, y_linha))
    valores.append(converte_eq_em_naosimbolico(sol_real_sympy, conjunto_x))
    valores.append(gen_seg_ord_alfa(y_zero, x_zero, h, num_repet, y_linha, 1 / 3))
    valores.append(gen_seg_ord_alfa(y_zero, x_zero, h, num_repet, y_linha, 1 / 4))
    valores.append(dormand_price_fixo(y_zero, x_zero, h, num_repet, y_linha))
    valores.append(dormand_price_adap(y_zero, x_zero, h, num_repet, contador))

    grafico(valores, y_novo, contador)
    print_dados(valores, titulo, y_linha, contador, y_zero, x_zero, h, num_repet)


def grafico(valores, titulo, contador):
    """
    Faz um gráfico dos métodos numéricos para solução de EDOs e os salva no diretorio
    :param valores: valores retornados da função "calcula"
    :param titulo: EDO a ser utilizada
    :param contador: contador para ser salvo no nome do gráfico
    """
    plt.style.use('seaborn')

    plot_edos, edos = plt.subplots()

    edos.plot(valores[0], valores[4], label='Função real', linewidth=4, alpha=1, color='black')
    edos.plot(valores[0], valores[1], label='Euler', linestyle=':', linewidth=3, color='#FE4A49', alpha=0.9)
    edos.plot(valores[0], valores[2], label='Euler melhorado', linestyle=':', linewidth=3, color='#2AB7CA', alpha=0.8)
    edos.plot(valores[0], valores[3], label='Euler modficado', linestyle=':', linewidth=3, color='#FEC620', alpha=0.7)
    edos.plot(valores[0], valores[5], label='Genérico de segunda ordem com alfa = 1/3', linewidth=3, alpha=0.6,
              linestyle=':', color='#1A5274')
    edos.plot(valores[0], valores[6], label='Genérico de segunda ordem com alfa = 1/4', linewidth=3, alpha=0.5,
              linestyle=':', color='#B892FF')
    edos.plot(valores[0], valores[7], label='Dormand Price com passo fixo', alpha=0.4, linewidth=3,
              linestyle=':', color='#BAFF29')
    edos.plot(valores[8][0], valores[8][1], label='Dormand Price com passo adaptativo', alpha=0.3, linewidth=3,
              linestyle=':')

    edos.legend(fontsize='medium')
    edos.set_title(f"Métodos numéricos para solução gráfica da seguinte EDO: y' = {titulo}")

    plt.savefig(f'Métodos numéricos para solução gráfica {contador}')


def print_dados(valores, titulo, y_linha, contador, y_zero, x_zero, h, num_repet):
    """
    Imprime os valores calculados e seus respectivos erros em tabelas, estes erros são enviados para função
    grafico_erros() para que os erros dos métodos numéricos sejam plotados
    :param valores: valores retornados da função "calcula"
    :param titulo: String com a função y(usado para o título dos gráficos e tabelas)
    :param y_linha: y' simbolico
    :param contador: contador
    :param y_zero: y inicial
    :param x_zero: x inicial
    :param h: passo
    :param num_repet: número de repetições do passo
    """
    print('')
    print(f'#################################### Problema 1 - {contador} #########################################')
    print('')
    print(f'       EDO: dy/dx = {y_linha}')
    print('')
    print(f"       Cond. Iniciais: y0 = {y_zero}   x0 = {x_zero}   h = {h}")
    print('')
    print(f"       S. Exata:  y(x) = {titulo}")
    print('')
    print('------------------------------------- TABELA DE RESULTADOS -------------------------------------')
    print('|=======|===========|===========|===========|===========|============|============|============|')
    print('|   X   |  S.Exata  |   Euler   |  E. Mel.  |  E. Mod.  | 2ord a=1/3 | 2ord a=1/4 | ODE45 fixo |')
    print('|=======|===========|===========|===========|===========|============|============|============|')

    for i in range(num_repet + 1):

        if (valores[4][i] and valores[1][i] and valores[2][i] and valores[3][i] and valores[6][i]) < 10:

            print(f'| {valores[0][i]:.3f} | {valores[4][i]:.7f} | {valores[1][i]:.7f} | {valores[2][i]:.7f} | '
                  f'{valores[3][i]:.7f} | {valores[5][i]:.7f}  | {valores[6][i]:.7f}  | {valores[7][i]:.7f}  |')
        else:

            print(f'| {valores[0]:.3f} | {valores[4][i]:.6f} | {valores[1][i]:.6f} | {valores[2][i]:.6f} | '
                  f'{valores[3][i]:.6f} | {valores[5][i]:.6f}  | {valores[6][i]:.6f}  | {valores[7][i]:.6f}  |')

    print('|=======|===========|===========|===========|===========|============|============|============|')
    print('')
    print('')
    print('------------------------------------- TABELA DE ERROS --------------------------------------')
    print('|=======|===========|===========|===========|============|============|====================|')
    print('|   X   |   Euler   |  E. Mel.  |  E. Mod.  | 2ord a=1/3 | 2ord a=1/4 |      ODE45 fixo    |')
    print('|=======|===========|===========|===========|============|============|====================|')

    #   erros é uma lista de listas contendo [erro_euler, erro_euler_mel, erro_euler_mod, erro_gen_a1, erro_gen_a2]
    erro_euler = []
    erro_euler_mel = []
    erro_euler_mod = []
    erro_gen_a1 = []
    erro_gen_a2 = []
    erro_dp_fixo = []
    for i in range(num_repet + 1):

        erro_euler.append(abs(valores[1][i] - valores[4][i]))
        erro_euler_mel.append(abs(valores[2][i] - valores[4][i]))
        erro_euler_mod.append(abs(valores[3][i] - valores[4][i]))
        erro_gen_a1.append(abs(valores[5][i] - valores[4][i]))
        erro_gen_a2.append(abs(valores[6][i] - valores[4][i]))
        erro_dp_fixo.append(abs(valores[7][i] - valores[4][i]))

        if (erro_euler[i] and erro_euler_mel[i] and erro_euler_mod[i] and erro_gen_a1[i] and erro_gen_a2[i]) < 10:

            print(f'| {valores[0][i]:.3f} | {erro_euler[i]:.7f} | {erro_euler_mel[i]:.7f} | '
                  f'{erro_euler_mod[i]:.7f} | {erro_gen_a1[i]:.7f}  | {erro_gen_a2[i]:.7f}  | {erro_dp_fixo[i]:.15f}  |')
        else:

            print(f'| {valores[0][i]:.3f} | {erro_euler[i]:.6f} | {erro_euler_mel[i]:.6f} | '
                  f'{erro_euler_mod[i]:.6f} | {erro_gen_a1[i]:.6f}  | {erro_gen_a2[i]:.6f}  | {erro_dp_fixo[i]:.15f}  |')

    print('|=======|===========|===========|===========|============|============|====================|')
    print('')

    # erros é uma lista de listas contendo [erro_euler, erro_euler_mel, erro_euler_mod, erro_gen_a1, erro_gen_a2,
    # erro_dp_fixo]
    erros = [erro_euler, erro_euler_mel, erro_euler_mod, erro_gen_a1, erro_gen_a2, erro_dp_fixo]
    grafico_erros(erros, valores[0], contador, titulo)


def grafico_erros(erros, conjunto_x, contador, titulo):
    """
    Faz um gráfico dos erros dos métodos numéricos para solução de EDOs com relação aos valores da solução análitica
    da EDO
    :param erros: conjunto de erros de cada método numérico com relação aos valores da solução análitica da EDO
    :param conjunto_x: conjunto de x dos erros
    :param contador: contador
    :param titulo: EDO a ser utilizada
    """
    aux_euler = []
    aux_euler_mel = []
    aux_euler_mod = []
    aux_2gen_umterco = []
    aux_2gen_umquarto = []
    aux_dp_fixo = []
    for i in range(len(erros[0])):
        if (erros[0][i] or erros[1][i] or erros[2][i] or erros[3][i] or erros[4][i]) == 0:
            i += 1

        aux_euler.append(round(ln(erros[0][i]), 15))
        aux_euler_mel.append(round(ln(erros[1][i]), 15))
        aux_euler_mod.append(round(ln(erros[2][i]), 15))
        aux_2gen_umterco.append(round(ln(erros[3][i]), 15))
        aux_2gen_umquarto.append(round(ln(erros[4][i]), 15))
        aux_dp_fixo.append(round(ln(erros[5][i]), 15))

    #   erros_ln é uma lista de listas contendo os ln dos valores da lista de listas erros
    erros_ln = [aux_euler, aux_euler_mel, aux_euler_mod, aux_2gen_umterco, aux_2gen_umquarto, aux_dp_fixo]

    plt.style.use('seaborn')

    plot_edos, edos_erros = plt.subplots()

    edos_erros.plot(conjunto_x, erros_ln[0], label='Erro Euler', linewidth=2, marker='o', color='#FE4A49')
    edos_erros.plot(conjunto_x, erros_ln[1], label='Erro Euler melhorado', linewidth=2, marker='v', color='#2AB7CA')
    edos_erros.plot(conjunto_x, erros_ln[2], label='Erro Euler modificado', linewidth=2, marker='^', color='#FEC620')
    edos_erros.plot(conjunto_x, erros_ln[3], label='Erro Genérico de segunda ordem com alfa = 1/3', linewidth=2,
                    marker='s', color='#1A5274')
    edos_erros.plot(conjunto_x, erros_ln[4], label='Erro Genérico de segunda ordem com alfa = 1/4', linewidth=2,
                    marker='*', color='#B892FF')
    edos_erros.plot(conjunto_x, erros_ln[5], label='Dormand Price com passo fixo', linewidth=2,
                    marker='>', color='#BAFF29')

    edos_erros.legend(loc=(0, 0.2), fontsize='x-small', framealpha=1)
    edos_erros.set_title(f"Erros de cada método em relação ao y(x) = {titulo}")
    edos_erros.set_ylabel(f'ln(erros)')

    plt.savefig(f'Erros em escala logaritmica dos Métodos numéricos para solução gráfica {contador}')
