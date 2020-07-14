import math
import numpy as np

class Neurona:
    def __init__(self, input = list(), output = list(), de_input = False):
        self.input = list()
        self.val = 0.0
        self.de_input = de_input
        self.input_pesos = list()
        # Si la lista de input está vacía, entonces se queda ahí
        for i in input:
            self.agrega_input(i)
        self.output = list()
        # Si la lista de output está vacía, entonces se queda ahí
        for i in output:
            self.agrega_output(i)
        # Se le inicializa con un sesgo aleatorio
        self.sesgo = np.random.uniform()

    """ 
        Calcula y retorna la suma ponderada de la capa anterior (input)
            sumado al sesgo.
    """
    def pesos(self):
        suma = 0
        for i in range(0, len(self.input)):
            suma += self.input[i].get_val() * self.input_pesos[i]
        return suma + self.sesgo


    """
        Retorna el valor de la neurona, 
            ya sea el activado o el directo en caso de ser de la capa de input
    """
    def get_val(self):
        # Si es capa oculta, se calcula desde las capas de input
        if not self.de_input:
            return self.sigmoid()

        # Sino, se pasa el valor que tiene
        else:
            return self.val
            

    """
        Cambia el valor que guarda la neurona de input
    """
    def set_val(self, val):
        if self.de_input:
            self.val = val
        else:
            raise ANNError(val, "No se puede establecer valor a capa que no sea input.")


    """
        Función de activación, 
            usa sigmoide sobre la suma ponderada de los pesos
    """
    def sigmoid(self):
        return 1 / (1 + math.exp(-self.pesos()))


    """
        Agrega una neurona como input, 
            inicializando con un peso aleatorio
    """
    def agrega_input(self, neurona):
        self.input.append(neurona)
        # Se le inicializa con un peso aleatorio
        self.input_pesos.append(np.random.uniform())

    """
        Agrega una neurona al output
    """
    def agrega_output(self, neurona):
        self.output.append(neurona)

    """
        Agrega una capa de neuronas al output
    """
    def agrega_capa_out(self, neuronas):
        for neurona in neuronas:
            self.output.append(neurona)


    def print(self):
        print(self.get_val())


class ANNError(Exception):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


"""
    La red neuronal, tiene una lista de capas que son listas de neuronas.
"""
class ANN:
    
    """
        Constructor por defecto:
            shape: arreglo que define la cantidad de capas y neuronas por capa de la red.
                    Cada espacio representa una capa, 
                        y el número de este representa la cantidad de neuronas que tiene.
    """
    def __init__(self, shape = [4,3,3,4]):

        # Se necesitan al menos capa de input y output para que sea una red válida
        if len(shape) < 2:
            raise ANNError(shape, "La red necesita al menos 2 capas para funcionar")
        

        # Se confirma que la forma que se pasó por parámetro sea válida
        for i in range(0, len(shape)):
            if not isinstance(shape[i], int):
                raise ANNError(shape[i], "El valor debe ser un entero")
            if shape[i] < 1:
                raise ANNError(shape[i], "La cantidad de elementos por capa debe ser > 0")

        self.capas = list()
        # Se crean las capas
        for i in range(0, len(shape)):
            aux = list()
            # Se crean neuronas por capa
            for j in range(0, shape[i]):
                # Se crea una neurona con el input de las capas anteriores

                # Si es la primera capa, no tiene capa de input
                if i == 0:
                    auxN = Neurona(de_input=True)
                else:
                    auxN = Neurona(self.capas[i-1])

                aux.append(auxN)
            self.capas.append(aux)
                

        # Se conecta cada capa con su sucesora
        for i in range(len(shape)-2, -1, -1):
            for j in range(0, len(self.capas[i])):
                self.capas[i][j].agrega_capa_out(self.capas[i+1])
        
    def imprime(self):
        for i in range(0, len(self.capas)):
            print("Capa ", str(i))
            for j in range(0, len(self.capas[i])):
                self.capas[i][j].print()


    """
        Calcula una predicción de los valores de input:
            input: lista de valores de entrada.
    """
    def predict(self, input):
        salida = list()
        if len(input) != len(self.capas[0]):
            raise ANNError("len(input) = " + str(len(input)), "La cantidad de elementos de entrada debe ser igual a " + str(len(self.capas[0])))
        # Se cambia el valor de la capa de input
        for i in range(0, len(self.capas[0])):
           self.capas[0][i].set_val(input[i])

        # Se predice el valor
        for i in range(0, len(self.capas[len(self.capas)-1])):
            salida.append(self.capas[len(self.capas)-1][i].get_val())
        return salida

def main():
    rnn = ANN(shape=[4, 3, 3, 3])
    rnn.imprime()
    val = rnn.predict([400, 3, 400000, 40000])
    print(val)

if __name__ == "__main__":
    main()