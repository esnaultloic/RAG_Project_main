🚀 CV RAG Assistant (LlamaIndex + Gemini)

Assistant RAG déployé qui répond aux questions des recruteurs à partir de documents PDF (CV + formation/certifications). Le système utilise LlamaIndex (pipeline RAG), ChromaDB (vecteurs persistants) et Google Gemini (embeddings + génération).

### Caractéristiques
- Récupération sémantique sur les PDF placés dans `data/`
- LLM: Gemini 2.5 Flash, Embeddings: text-embedding-004
- Stockage: ChromaDB (dossier `./chroma_db`)
- Application Web: Streamlit (`app.py`)
- Top-k fixé à 4 pour des réponses robustes

### Variables d’environnement

**Pour le développement local :**
Créer un fichier `.env` à la racine (ce fichier est ignoré par Git) :
```
GEMINI_API_KEY="VOTRE_CLE_API"
```

**Pour Streamlit Cloud :**
1. Allez dans les paramètres de votre app Streamlit Cloud
2. Section "Secrets" → ajoutez :
   ```toml
   GEMINI_API_KEY = "votre_cle_api"
   ```
3. Redéployez l'app

### Lancer en local (optionnel)
```
pip install -r requirements.txt
py -m streamlit run app.py
```
Ou:
```
python -m streamlit run app.py
```

### Déploiement
- Streamlit Cloud: connectez le dépôt, définissez la variable `GEMINI_API_KEY`, ciblez `app.py`. L’app reste accessible en ligne sans lancer de commande locale.
- Heroku/Render (alternatif): `Procfile` fourni (`web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`). Pensez à configurer `GEMINI_API_KEY` côté plateforme.

### Structure principale
- `rag_pipeline.py`:
  - `setup_rag_index()`: crée/charge l’index RAG (chunking via `SentenceSplitter`, ChromaDB persistant)
  - `query_rag(prompt, ..., similarity_top_k=4)`: interroge l’index et génère la réponse
- `app.py`: interface Streamlit épurée adaptée aux recruteurs
- `data/`: placez vos PDF (CV, formation, certifications)

### Utilisation
1) Placez vos PDF dans `data/`
2) Ouvrez l’app (déployée ou locale) et posez votre question
3) La réponse est ancrée dans le contenu des documents

### Notes
- L’app est pensée pour un usage public; aucune option technique n’est exposée.
- Si vous mettez à jour les documents, redéployez ou relancez l’app pour reconstruire l’index si nécessaire.