"""
Script d'importation automatique du dashboard Grafana DNS Shield
Utilise l'API Grafana pour importer le dashboard JSON
"""

import json
import os
import requests
from pathlib import Path

# Configuration
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
GRAFANA_USER = os.getenv("GRAFANA_USER", "admin")
GRAFANA_PASSWORD = os.getenv("GRAFANA_PASSWORD", "admin")
DASHBOARD_JSON_PATH = Path(__file__).parent.parent / "config" / "grafana" / "dashboards" / "dns-shield-overview.json"

def import_dashboard():
    """Importe le dashboard dans Grafana via l'API"""
    
    print("Importation du dashboard Grafana DNS Shield...")
    print(f"Fichier: {DASHBOARD_JSON_PATH}")
    
    # Charger le JSON du dashboard
    try:
        with open(DASHBOARD_JSON_PATH, 'r', encoding='utf-8') as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"Erreur: Fichier {DASHBOARD_JSON_PATH} introuvable!")
        return False
    except json.JSONDecodeError as e:
        print(f"Erreur JSON: {e}")
        return False
    
    # Préparer la payload pour l'API Grafana
    payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "message": "Auto-imported via script"
    }
    
    # Envoyer à l'API Grafana
    try:
        response = requests.post(
            f"{GRAFANA_URL}/api/dashboards/db",
            json=payload,
            auth=(GRAFANA_USER, GRAFANA_PASSWORD),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            dashboard_url = f"{GRAFANA_URL}{result.get('url', '')}"
            print("Dashboard importé avec succès!")
            print(f"URL: {dashboard_url}")
            return True
        else:
            print(f"Erreur {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("Impossible de se connecter à Grafana!")
        print("Vérifiez que Grafana tourne sur http://localhost:3000")
        return False
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        return False

def verify_prometheus():
    """Vérifie que Prometheus collecte des données"""
    
    print("\nVérification de Prometheus...")
    
    try:
        # Vérifier que Prometheus répond
        response = requests.get("http://localhost:9090/api/v1/query", 
                               params={"query": "dns_queries_total"},
                               timeout=5)
        
        if response.status_code != 200:
            print("Prometheus ne répond pas correctement")
            return False
        
        data = response.json()
        results = data.get('data', {}).get('result', [])
        
        if not results:
            print("Prometheus fonctionne mais aucune métrique dns_queries_total trouvée")
            print("Lancez du trafic: python scripts/simulate_traffic.py --rate 5 --duration 120")
            return False

        # Afficher les métriques trouvées
        print("Prometheus collecte des données:")
        for result in results:
            service = result['metric'].get('service', 'unknown')
            value = result['value'][1]
            print(f"   • {service}: {value} requêtes")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("Impossible de se connecter à Prometheus!")
        print("Vérifiez que Prometheus tourne sur http://localhost:9090")
        return False
    except Exception as e:
        print(f"Erreur: {e}")
        return False

def main():
    """Point d'entrée principal"""
    
    print("="*60)
    print("DNS SHIELD - IMPORTATION DASHBOARD GRAFANA")
    print("="*60)
    print()

    # Vérifier Prometheus d'abord
    if not verify_prometheus():
        print("\nContinuez quand même? Le dashboard sera vide sans données.")
        response = input("Continuer (o/n)? ")
        if response.lower() != 'o':
            print("Annulé.")
            return
    
    # Importer le dashboard
    print()
    if import_dashboard():
        print("\n" + "="*60)
        print("IMPORTATION TERMINÉE AVEC SUCCÈS!")
        print("="*60)
        print("\nProchaines étapes:")
        print("   1. Ouvrir http://localhost:3000")
        print("   2. Chercher 'DNS Shield - Monitoring'")
        print("   3. Profiter des 21 visualisations en temps réel!")
        print()
    else:
        print("\n" + "="*60)
        print("ÉCHEC DE L'IMPORTATION")
        print("="*60)
        print("\nImport manuel:")
        print("   1. Ouvrir http://localhost:3000")
        print("   2. Menu + → Import")
        print("   3. Upload JSON file")
        print(f"   4. Sélectionner: {DASHBOARD_JSON_PATH}")
        print()

if __name__ == "__main__":
    main()
