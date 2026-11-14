"""Original best-checkpoint comparison rules, including ties."""




def check_and_save(model, dataset, result, check, save_model=True, parallel=False):
    file_name = 'best_'
    for key in check.keys():
        if key == 'MAE' or key == 'Loss':
            if check[key] >= result[key]:
                if save_model:
                    if parallel:
                        model.module.save_model(file_name + key, dataset)
                    else:
                        model.save_model(file_name + key, dataset)
                check[key] = result[key]
        else:
            if check[key] <= result[key]:
                if save_model:
                    if parallel:
                        model.module.save_model(file_name + key, dataset)
                    else:
                        model.save_model(file_name + key, dataset)
                check[key] = result[key]
    return check


class best_result:
    def __init__(self):
        self.result_save = {'Has0_acc_2': {},
                            'Has0_F1_score': {},
                            'Non0_acc_2': {},
                            'Non0_F1_score': {},
                            'Mult_acc_5': {},
                            'Mult_acc_7': {},
                            'MAE': {},
                            'Corr': {},
                            'Loss': {}}
        self.result_check = {'Has0_acc_2': 0,
                             'Has0_F1_score': 0,
                             'Non0_acc_2': 0,
                             'Non0_F1_score': 0,
                             'Mult_acc_5': 0,
                             'Mult_acc_7': 0,
                             'MAE': 100,
                             'Corr': 0,
                             'Loss': 10000}

    def check(self, result, result_test):
        for key in result.keys():
            if key == 'MAE' or key == 'Loss':
                if self.result_check[key] > result[key]:
                    self.result_check[key] = result[key]
                    self.result_save[key] = result_test
            else:
                if self.result_check[key] < result[key]:
                    self.result_check[key] = result[key]
                    self.result_save[key] = result_test

    def print(self):
        for key in self.result_save.keys():
            print('==========================')
            print('Best test based on %s' % key)
            for key2 in self.result_save[key].keys():
                print('\t%s: %s' % (key2, self.result_save[key][key2]))
