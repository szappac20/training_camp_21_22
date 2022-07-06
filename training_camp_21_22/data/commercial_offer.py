class CommercialOffer(object):
    def __init__(self, price_dict, name):

        self.price_dict = price_dict
        self.name = name

    def compute_cost(self, bill):
        cost_1 = bill.f1 * self.price_dict["f1"]
        cost_2 = bill.f2 * self.price_dict["f2"]
        cost_3 = bill.f3 * self.price_dict["f3"]

        return cost_1 + cost_2 + cost_3

    def compute_yearly_cost(self, bills):
        cost_1 = bills.f1 * self.price_dict["f1"]
        cost_2 = bills.f2 * self.price_dict["f2"]
        cost_3 = bills.f3 * self.price_dict["f3"]

        return cost_1 + cost_2 + cost_3
