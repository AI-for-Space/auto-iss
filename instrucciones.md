# Spacex ISS Simulator
Ejecutar [SpaceX ISS Docking Simulator](https://iss-sim.spacex.com/) de forma local.

# Crear el entorno de simulación
1. Clonar el siguiente repositorio
    ```bash
    git clone https://github.com/matthewgiarra/spacex-iss-sim
    ```

2. Crear un servidor local
    ```bash
    cd spacex-iss-sim
    python3 -m http.server 5555
    ```

# Entrenar o testear el agente

Si se desea entrenar el agente simplemente ejecute el archivo main en la carpeta auto-iss-complex

Si se desea testear los dos agentes en el modelo mas simple introduxca sus dos agentes en la carpeta auto-iss-demo y ejecute el archivo main.ipnyb o el archivo play_demo.py.
