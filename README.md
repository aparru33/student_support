# student_support
Dashboard using AI to enable schools to prioritize students to be coached


# French exercice statement

## Énoncé
Face à la chute du niveau scolaire constaté à la suite de la fermeture des écoles, le “Ministério da Educação” (Ministère de l'Éducation portugais) vous contacte, avec l'idée d'utiliser la data et l’IA pour tenter de remédier à la situation.

Le Ministère souhaiterait que les conseillers pédagogiques de chaque établissement puissent disposer d’un outil leur permettant de prioriser les élèves à accompagner. Pour cela, ils imaginent un dashboard qui permettrait de prioriser les élèves à accompagner en fonction de la complexité et de la valeur d’un tel accompagnement. 

Ce dashboard pourrait par exemple se centrer autour d’un graphe permettant de visualiser l’ensemble des élèves de l’établissement suivant deux axes. Le premier axe présenterait la note actuelle de l’élève, indiquant ainsi l’intérêt qu’il y aurait à lui apporter un soutien personnalisé. Le deuxième axe permettrait d’évaluer la complexité d’accompagner l’élève pour améliorer son niveau scolaire (en se basant par exemple sur la présence d’indicateurs actionnables tels qu’un niveau d’absentéisme fort, la consommation d’alcool ou un temps d’étude hebdomadaire en dessous de la moyenne). Voici ce à quoi pourrait ressembler un tel dashboard (les élèves à aider en priorité correspondant aux points en haut à droite).

Il est bien sûr possible de proposer un autre type de visualisation si elle vous paraît plus pertinente que celle imaginée par le Ministère, et qu’elle n’est pas trop complexe à développer.

## Livrable attendu
Votre mission consiste à développer le code permettant de déployer une première version de cet outil sur l’infrastructure de l’établissement pilote. Cette première version devra être automatisable, pouvoir être déployée facilement et pouvoir être maintenue et améliorée dans le temps. 

Il n’est pas nécessaire de documenter l’outil ou de l’accompagner d’une présentation quelconque si son utilisation est suffisamment intuitive. Il n’est pas non plus demandé de trop s’attarder sur la partie front du dashboard, l’aspect visuel ou le contenu textuel. On pourra par exemple envisager d’utiliser des librairies telles que Streamlit, Dash ou Gradio.

Il n’est pas aussi utile de documenter l’approche imaginée pour la mise en place ou l’utilisation de l’outil. On pourra en discuter durant l’entretien.

## Données à disposition
Afin de vous aider dans cette tâche, vous vous appuierez sur les résultats scolaires en mathématiques des étudiants d’une école pilotes, ainsi que sur les réponses à un questionnaire qui leur a été transmis préalablement : 

 - Dataset 
 - Description des données

## Critères d’évaluation
- Intérêt de la solution proposée pour l’utilisateur. Dans le cadre de cet exercice, il n’est pas nécessaire que le dashboard soit très esthétique ou qu’il contienne énormément de visualisations ; l’important est que les visualisations réalisées et les données fournies soient pertinentes pour l’utilisateur.
- Facilité de mise en place de la solution proposée (facilité à déployer et à lancer la solution proposée sur tous types de serveurs).
- Facilité d’amélioration de la solution proposée (facilité pour un nouveau développeur à prendre en main le code, à comprendre son fonctionnement et les raisons des choix qui ont été faits ; facilité à améliorer la solution sans introduire de bugs ; etc).
- Pragmatisme et pertinence de la démarche proposée à travers le dashboard.