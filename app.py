# app.py - Lanceur de microservices pour DNS Shield

import subprocess
import time
import sys
import os

# Définir les services à lancer
# Chaque service est un dictionnaire avec le chemin du script et le port attendu
SERVICES = [
    {"name": "DGA Detector", "path": "src/dga_detector.py", "port": 8001},
    {"name": "BERT Service", "path": "src/bert_service.py", "port": 8002},
    {"name": "Ensemble ML Service", "path": "src/ensemble_ml.py", "port": 8003},
    {"name": "API Gateway", "path": "src/api_gateway.py", "port": 9000},
]

def main():
    """
    Lance tous les microservices DNS Shield en tant que sous-processus.
    """
    print("=" * 40)
    print("Démarrage des microservices DNS Shield...")
    print("=" * 40)
    
    processes = []

    # Utiliser l'exécutable Python de l'environnement actuel
    python_executable = sys.executable
    print(f"Utilisation de l'interpréteur Python : {python_executable}\n")

    for service in SERVICES:
        script_path = service["path"]
        
        if not os.path.exists(script_path):
            print(f"ERREUR: Le script pour le service '{service['name']}' est introuvable à '{script_path}'.")
            continue

        print(f"Démarrage du service : {service['name']} sur le port {service['port']}...")
        
        try:
            # Lancer le script en tant que sous-processus non bloquant
            # `creationflags` est utilisé sous Windows pour éviter d'ouvrir des fenêtres de console
            if sys.platform == "win32":
                process = subprocess.Popen(
                    [python_executable, script_path],
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                process = subprocess.Popen([python_executable, script_path])
                
            processes.append(process)
            print(f"Le service '{service['name']}' a démarré avec le PID : {process.pid}")
            time.sleep(2)  # Attendre un peu pour que le service s'initialise
        
        except Exception as e:
            print(f"ERREUR lors du démarrage du service '{service['name']}': {e}")

    print("\n" + "=" * 40)
    print("Tous les services ont été lancés.")
    print("La passerelle API devrait être disponible sur le port 9000.")
    print("Appuyez sur CTRL+C dans ce terminal pour arrêter tous les services.")
    print("=" * 40)

    try:
        # Garder le script principal en vie pour pouvoir intercepter CTRL+C
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterruption reçue. Arrêt de tous les services...")
        for process in processes:
            process.terminate()  # Envoyer le signal de terminaison
        
        # Attendre que les processus se terminent
        for process in processes:
            process.wait()
            
        print("Tous les services ont été arrêtés. Au revoir!")

if __name__ == "__main__":
    main()
