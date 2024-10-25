# API de prédiction de sentiments

## Objectifs

L’objectif de cette API de prédiction de sentiments est de permettre à "Air Paradis" d’anticiper les réactions des utilisateurs sur les réseaux sociaux en analysant automatiquement les sentiments des tweets. En fournissant une prédiction binaire (positif ou négatif) pour chaque tweet, cette API vise à détecter rapidement les sentiments négatifs, aidant ainsi l’entreprise à réagir avant qu'un bad buzz potentiel ne se propage.

L'API est conçue pour être facilement intégrée dans des applications ou interfaces, où elle peut recevoir un texte en entrée (tweet) et retourner une prédiction de sentiment. Grâce à une approche MLOps, elle garantit aussi un suivi continu des performances et une gestion centralisée des modèles, avec un système de traçabilité pour l’amélioration progressive des prédictions.

## Découpage des dossiers
- embeddings/ : contient les fichiers de word embeddings utilisés pour la représentation vectorielle des tweets. Ce dossier stocke les modèles de texte préentraînés ou générés, qui sont utilisés pour transformer les tweets en données exploitables par les modèles de prédiction.
- models/ : héberge les fichiers de modèles de machine learning ou deep learning créés pour la prédiction des sentiments. Ce dossier permet de centraliser et de versionner les différents modèles utilisés ou testés durant le projet.
- static/ : contient les ressources statiques utilisées pour l'interface web, comme les fichiers JavaScript et CSS. Ce dossier permet de gérer et de personnaliser les éléments visuels et les comportements dynamiques de l’interface HTML.
- templates/ : inclut l’interface HTML de l’application, utilisée pour visualiser et interagir avec l’API. Ce dossier est structuré de manière à suivre la convention Flask pour le rendu des templates HTML.
- Fichiers à la racine :
  - app.py : fichier principal qui initialise et configure l'API Flask pour la prédiction de sentiments. Il inclut les routes pour recevoir un tweet et retourner une prédiction de sentiment (positif ou négatif).
  - requirements.txt : liste de tous les packages Python nécessaires pour exécuter le projet, facilitant la reproduction de l’environnement d’exécution.
  - test_app.py : contient les tests unitaires, écrits avec la bibliothèque unittest, pour vérifier le bon fonctionnement des routes et des prédictions de l'API.
