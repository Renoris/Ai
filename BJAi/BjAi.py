from classification_util import ClassificationUtil

busot=ClassificationUtil()
busot.ignore_warning()
busot.read('mushrooms.csv')


busot.drop(['cap-shape','bruises','gill-attachment','gill-spacing','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-color-above-ring','stalk-surface-below-ring','stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type','spore-print-color','population','habitat'])
busot.show()
busot.heatmap()
busot.myviolinplot('class','cap-color')
print('svm:')
busot.run_svm(['cap-color','cap-surface','odor','gill-size','gill-color'],'class')
print('neighbor:')
busot.run_neighbor_classifier(['cap-color','cap-surface','odor','gill-size','gill-color'],'class',3)
print('logistic:')
busot.run_logistic_regression(['cap-color','cap-surface','odor','gill-size','gill-color'],'class')
print('decision:')
busot.run_decision_tree_classifier(['cap-color','cap-surface','odor','gill-size','gill-color'],'class')