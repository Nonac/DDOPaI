mport torch.nn as nn

from prettytable import PrettyTable


class base_model(nn.Module):

    def __init__(self):
        super(base_model, self).__init__()
        self.model_type = None
        self.plan = None
        self.remaining_params, self.total_params = 1, 1
        self.weight_ratio = self.remaining_params / self.total_params * 100
        self.table = None
        self.train_trainset_loss_arr = [0]
        self.train_testset_loss_arr = [0]
        self.train_testset_accuracy_arr = [0]
        self.val_loss_arr = []
        self.val_accuracy_arr = []

    def weight_counter(self):
        """
        This function Search nn.Linear or nn.Conv2D module and calculates the number of 1s in the mask.
        save the result as table(by prettytable).
        """
        self.table_initialize()
        self.remaining_params, self.total_params = 0, 0

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) is True:
                self.check_mask_weight(name, module)

        self.weight_ratio = self.remaining_params / self.total_params * 100
        self.table.add_row(['*total*',
                            f"{self.total_params:.0f}",
                            f"{self.remaining_params:.0f} / {self.total_params-self.remaining_params:.0f}",
                            f"{self.weight_ratio:.2f}"])

    def check_mask_weight(self, name, module):
        """
        Find mask in modeule.named_buffers(). And after counting the number of 1 in the parameter, record it in the
        table.
        INPUT:
            opt(:obj:`str`):
                Name of module.
            opt(:obj:`nn.Module`):
                Module(nn.Linear of nn.Conv2d) with mask.
        """
        for buf_name, buf_param in module.named_buffers():
            if "weight_mask" in buf_name:
                remaing_p = buf_param.detach().cpu().numpy().sum()
                total_p = buf_param.numel()
                self.remaining_params += remaing_p
                self.total_params += total_p
                self.table.add_row([name, f"{total_p:.0f}",
                                    f"{remaing_p:.0f} / {total_p-remaing_p:.0f}",
                                    f"{remaing_p/total_p*100:.2f}"])

    def get_teacher_plan_base(self):
        """
        """
        plan_base = []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) is True:
                remaining_params, total_params = self.mask_weight_for_st(module)
                plan_base.append((remaining_params, total_params))
        return plan_base

    def mask_weight_for_st(self, module):
        """
        """
        remaining_params, total_params = 0, 0
        for buf_name, buf_param in module.named_buffers():
            if "weight_mask" in buf_name:
                remaining_params += buf_param.detach().cpu().numpy().sum()
                total_params += buf_param.numel()

        return remaining_params, total_params

    def get_student_plan(self, plan_base):

        plans = {
            'vgg11': [0, 'M', 0, 'M', 0, 0, 'M', 0, 0, 'M', 0, 0],
            'vgg13': [0, 0, 'M', 0, 0, 'M', 0, 0, 'M', 0, 0, 'M', 0, 0],
            'vgg16': [0, 0, 'M', 0, 0, 'M', 0, 0, 0, 'M', 0, 0, 0, 'M', 0, 0, 0],
            'vgg19': [0, 0, 'M', 0, 0, 'M', 0, 0, 0, 0, 'M', 0, 0, 0, 0, 'M', 0, 0, 0, 0]
        }

        plan = plans[self.model_type]

        if "vgg" in self.model_type:
            plan_init = plan_base.pop(0)
            plan[0] = round(plan_init[0] / (9 * 3))
            plan_temp = plan[0]

            for layer_info in enumerate(plan):
                if layer_info[1] == 0:
                    plan_data = plan_base.pop(0)
                    plan[layer_info[0]] = round(plan_data[0]/(plan_temp*3))
                    plan_temp = plan[layer_info[0]]
        else:
            plan = [0]

        return plan

    def get_student_plan(self):
        plan_base = self.get_teacher_plan_base()
        plan = self.get_student_plan(plan_base)
        return plan

    def table_initialize(self):
        """
        Prettytable initializer.
        """
        self.table = PrettyTable(['Layer', 'Total_Weight', 'Remaining/Pruned', 'Ratio(%)'])
        self.table.align["Layer"] = "l"
        self.table.align["Total_Weight"] = "l"
        self.table.align["Remaining/Pruned"] = "l"
        self.table.align["Ratio(%)"] = "l"

    def logging_table(self, logger):
        """
        Log the table
        INPUT:
            logger(:obj:`logging.RootLogger`):
                RootLogger variables.
        """
        logger.logger.info(self.table)