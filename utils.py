import numpy as np

def inversa_generalizada(G,d):
    """
    Função para calcular a solução inversa generalizada.
    
    d = Gw
    
    Parâmetros:
    G : Matriz de design (dados).
    d : Vetor de dados observados.
        
    Retorna:
    w : Parâmetros do modelo estimados.
    """
    # Calculando a solução inversa generalizada
    w = np.linalg.inv(G.T @ G) @ G.T @ d
    return w

def preditor_linear(G, w):
    """
    Função para estimar a aproximação de do vetor de dados d
    
    Parâmetros:
    G : Matriz de design (dados).
    w : Parâmetros do modelo.
        
    Retorna:
    d_calc : Dado calculado.
    """
    d_calc = np.matmul(G,w)
    return d_calc

# Faça uma função para calcular o erro quadrático médio (EQM) entre o dado observado e o dado calculado
def erro_quadratico_medio(d, d_calc):
    """
    Função para calcular o erro quadrático médio (EQM) entre o dado observado e o dado calculado.
    
    Parâmetros:
    d : Vetor de dados observados.
    d_calc : Vetor de dados previstos.
        
    Retorna:
    erro : float
       eqm : erro quadrático médio.
    """

    return np.sum((d - d_calc)**2) / len(d)

def preditor_gardner(v, w):
    """
    Função para estimar a densidade (RHOB) a partir da velocidade (v) usando a relação de Gardner.
    
    Parâmetros:
    v : Vetor de velocidades (m/s).
    w : Parâmetros do modelo (w0 e w1).
        
    Retorna:
    rhob_calc : numpy.ndarray
        Densidade prevista (g/cm³).
    """
    rhob_calc = 10**w[0] * (v ** w[1])
    return rhob_calc

def inversa_generalizada_reg(G, d, lambda_reg=0.1):
    """
    Função para calcular a solução inversa generalizada com regularização.
    
    Parâmetros:
    G : Matriz de dados.
    d : Vetor de dados observados.
    lambda_reg : Termo de regularização.
        
    Retorna:
    w : Parâmetros do modelo estimados.
    """
    # Calculando a solução inversa generalizada com regularização
    w = np.matmul(np.matmul(np.linalg.inv(np.matmul(G.T,G) + lambda_reg**2 * np.eye(G.shape[1])),G.T),d)
    return w