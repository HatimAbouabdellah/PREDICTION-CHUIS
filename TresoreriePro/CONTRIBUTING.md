# Guide de contribution

Merci de votre intérêt pour le projet TresoreriePro ! Nous apprécions votre volonté de contribuer. Voici comment vous pouvez nous aider à améliorer ce projet.

## Comment contribuer

### Signaler un bug
1. Vérrez d'abord si le bug n'a pas déjà été signalé en consultant les [issues](https://github.com/votre-utilisateur/tresorerie-pro/issues).
2. Si ce n'est pas le cas, créez une nouvelle issue avec une description claire du problème.
3. Incluez des étapes pour reproduire le bug, le résultat attendu et le résultat obtenu.
4. Ajoutez des captures d'écran si nécessaire.

### Proposer une amélioration
1. Vérrez d'abord si l'amélioration n'a pas déjà été proposée.
2. Créez une issue pour discuter de votre proposition avant de commencer à coder.
3. Une fois approuvée, vous pouvez soumettre une pull request.

### Soumettre une pull request
1. Forkez le dépôt et créez une branche pour votre fonctionnalité :
   ```
   git checkout -b ma-nouvelle-fonctionnalite
   ```
2. Effectuez vos modifications et testez-les soigneusement.
3. Assurez-vous que votre code respecte les normes de style (voir ci-dessous).
4. Soumettez une pull request avec une description claire de vos modifications.

## Normes de code

### Style de code
- Suivez le [PEP 8](https://www.python.org/dev/peps/pep-0008/) pour le code Python.
- Utilisez des noms de variables et de fonctions descriptifs.
- Commentez votre code de manière claire et concise.

### Tests
- Assurez-vous que tous les tests passent avant de soumettre une pull request.
- Ajoutez des tests pour les nouvelles fonctionnalités.

### Documentation
- Mettez à jour la documentation pour refléter vos modifications.
- Ajoutez des docstrings aux nouvelles fonctions et classes.

## Environnement de développement

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/tresorerie-pro.git
   cd tresorerie-pro
   ```

2. Créez et activez un environnement virtuel :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. Installez les dépendances de développement :
   ```bash
   pip install -r requirements-dev.txt
   ```

## Processus de révision

1. Les pull requests seront examinées par les mainteneurs du projet.
2. Des modifications peuvent être demandées avant la fusion.
3. Une fois approuvées, vos modifications seront fusionnées dans la branche principale.

## Code de conduite

En participant à ce projet, vous acceptez de respecter le [Code de conduite](CODE_OF_CONDUCT.md).

## Remerciements

Merci de contribuer à faire de TresoreriePro un outil encore meilleur !
