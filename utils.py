import numpy as np

def inversa_generalizada(G,d):
    """
    Função para calcular a solução inversa generalizada.
    
    Parâmetros:
    G : numpy.ndarray
        Matriz de design (dados).
    d : numpy.ndarray
        Vetor de dados observados.
        
    Retorna:
    w : numpy.ndarray
        Parâmetros do modelo estimados.
    """
    # Calculando a solução inversa generalizada
    w = np.linalg.inv(G.T @ G) @ G.T @ d
    return w

#Crie uma função para realizar a predição dos dados com base em uma matriz de dados G e parâmetros w
def preditor(G, w):
    """
    Função para estimar a aproximação de do vetor de dados d
    
    Parâmetros:
    G : numpy.ndarray
        Matriz de design (dados).
    w : numpy.ndarray
        Parâmetros do modelo.
        
    Retorna:
    d_calc : numpy.ndarray
        Dados previstos.
    """
    d_calc = np.matmul(G,w)
    return d_calc