# YouTube Comments Scrapy pour les loulous
Récupérez les commentaires de **vidéos YouTube** spécifiques ou de **chaînes entières**. Fonctionne selon deux modes :
- **Mode sans API** (simple, sans clé API) : utilise des bibliothèques open source fiables pour récupérer les commentaires.
- **Mode API** (officiel, recommandé à grande échelle surtout pour du scrapy de masse) : utilise l'API YouTube Data v3 via de simples requêtes HTTPS avec votre clé API.
>**Important** : respectez toujours les conditions d'utilisation de YouTube et les lois locales. Utilisez le mode API pour une utilisation commerciale ou de production. Le mode sans API est destiné à un usage personnel ou éducatif et peut ne plus fonctionner si YouTube modifie son site.

## POUR COMMENCER AVEC CE POTIT OUTIL ( CLI WINDOWS )
1. **Téléchargez et décompressez** ce dossier quelque part (par exemple, `C:\yt_comments_tool`) rentrez dans le dossier avec un CMD WINDOWS EN **ADMINISTRATEUR**.

2. **Installez Python 3.9+** à partir de [python.org](https://www.python.org/downloads/windows/) si vous ne l'avez pas déjà.

3. Double-cliquez sur **`install.bat`** (crée `.venv` et installe les dépendances).

4. Double-cliquez sur **`run.bat`** pour afficher l'aide d'utilisation, exécutez à partir d'un terminal (CMD en mode administrateur) :
```cmd
   run.bat --help
   ```

## Exemples multiples maintenant
### A) Mode sans API (vidéo unique)
```cmd
run.bat --mode noapi --video https://www.youtube.com/watch?v=_pO7BRNdJdc&list=RD_pO7BRNdJdc&start_radio=1&ab_channel=JonPeck --max-comments 1000 --output-format csv 
```
**VIDEO TEST** NE **JUGEZ** PAS LA VIDEO DE TEST SVP
Génère le fichier `out/comments_pO7BRNdJdc.csv`.

### B) Mode sans API (chaîne entière)
Fonctionne avec **@handles**, `/channel/UC...`, `/c/username`, etc.
```cmd
run.bat --mode noapi --channel https://www.youtube.com/@loreal --max-comments 200 --since 2025-01-01 --output-format json
```
Cela affichera d'abord la liste des vidéos de la chaîne, puis récupérera les commentaires de chaque vidéo.

### C) Mode API (vidéo unique)
1) Obtenez une clé API auprès de Google Cloud → YouTube Data API v3.  
2) Exécutez ensuite :
```cmd
run.bat --mode api --api-key TA-CLEF-KEY --video https://www.youtube.com/watch?v=_pO7BRNdJdc&list=RD_pO7BRNdJdc&start_radio=1&ab_channel=JonPeck --include-replies --max-comments 1500 --output-format csv
```

### D) Mode API (chaîne par ID de chaîne uniquement)
Pour le mode API, préférez un lien `/channel/UC...` **ou** passez directement `--channel-id UCXXXXXXXXXX` **pour l'oréal** `--channel-id UCgmvz6qxtga4W6n1mfmM0Zw` | `--channel https://www.youtube.com/channel/UCjgOxNP15FxIYknHYgUBUM` :
```cmd
run.bat --mode api --api-key TA-CLEF-KEY --channel-id UCgmvz6qxtga4W6n1mfmM0Zw --since 2024-01-05 --max-comments 500
"run.bat --mode api --api-key TA-CLEF-API --channel-id UCgmvz6qxtga4W6n1mfmM0Zw --since 2024-01-05 --max-comments 500"
**l'oréal GLOBAL**

run.bat --mode api --api-key TA-CLEF-KEY --channel-id UCoC8wK7Hbj1saQUBlFVaJkA --since 2025-01-05 --max-comments 1000
run.bat --mode api --api-key TA-CLEF-KEY --channel-id UCjgOxNP15FxIYknHYgUBUMQ --since 2024-01-01 --max-comments 2000
**l'oréal PARIS USA**
```
> Astuce : si vous ne disposez que d'un identifiant tel que « https://www.youtube.com/@somecreator », passez en **mode sans API** ou recherchez l'ID de la chaîne et réutilisez-le.

## Sortie
- CSV ou JSON, une ligne par commentaire.
- Champs inclus : `commentId, videoId, videoTitle, channelId, channelTitle, author, authorChannelId, text, publishedAt, updatedAt, likeCount, replyCount, parentId, isReply`

Les fichiers sont enregistrés dans le dossier « out/ ».

## Traduction du CSV

- set DEEPL_API_KEY=VOTRE_CLEF_API
- arguments a utiliser dans le CLI = translate.bat --help (For help in all arguments) / --target-lang (EN,FR,EN-GB,DE,ES) / --source-lang (Detection auto si aucun input) / --deepl-key CLEF-API
```
translate.bat --inputs "out\comments__pO7BRNdJdc.csv" --target-lang FR
translate.bat --inputs out\comments_*.csv --target-lang FR --only-missing --batch-size 1000 --formality default
```

## Analyse de l'output
- `out/analysis/comments_labeled.csv` – chaque commentaire avec sa `primary_emotion` et `vader_compound`.
- `out/analysis/summary_emotions.csv` –  répartition par émotion..
- Graphique en PNG dans `out/analysis/`:
  - `emotions_distribution.png` (bar)
  - `emotions_by_video_topN.png` (stacked bar)
  - `emotions_over_time.png` (...)
  - `top_words_<emotion>.png` (optionel, et uniquement si il y'a assez de données)
- `out/analysis/marketing_report.xlsx` – Fichier Excel avec données étiquetées + résumés..

## Commande d'aide pour le CLI (command line interface)
toujours dans un terminal (CMD en mode administrateur)
```cmd
analyze.bat --help
```


## Utilisation avancée
```cmd
run.bat --mode noapi --video VIDEO_URL --max-comments 2000 --since 2025-01-01 --until 2025-09-01 --include-replies --output-format csv
```

## Remarques et limites
- Des **quotas API** s'appliquent lors de l'utilisation de l'API officielle.
- Le **mode sans API** utilise des données web publiques et peut être plus lent ; il peut cesser de fonctionner si YouTube modifie sa structure.
- Les grandes chaînes peuvent contenir de nombreuses vidéos ; envisagez d'utiliser `--max-videos` pour limiter
- comment avoir un **channel id**, en utilisant google chrome, faite un clic droit "Afficher le code source de la page" et cherchez "ChannelId" vous devriez trouver une ligne qui comme par "UC" (ex : UCjgOxNP15FxIYknHYgUBUMQ [L'oreal paris France])

**made with <3 by 550LY**

