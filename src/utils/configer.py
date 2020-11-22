import json
import os

class Configer(object):
    """Configuration details object

    Attributes:
        args (dict): Dictionary containing terminal parameters added to current procedure
        params (dict): Dictionary containing parameters in the json file provided

    """
    def __init__(self, args):
        """Configer constructor

        Args:
            args (argparse.Namespace): Object containing terminal parameters

        """
        self.args = args.__dict__
        self.params = None

        if not os.path.exists(args.hypes):
            raise ValueError('Json Path: {} not exists!'.format(args.hypes))

        json_stream = open(args.hypes, 'r')
        self.params = json.load(json_stream)
        json_stream.close()

    def get(self, *keys):
        """Item getter

        Args:
            *keys (list of str): List of keys

        Returns:
            el (str): Value retrived from args or params at keys location

        """
        if len(keys) == 0:
            return self.params

        key = keys[-1]
        if key in self.args and self.args[key] is not None:
            return self.args[key]

        el = self.params
        for key in keys:
            if key in el and el[key] is not None:
                el = el[key]
            else:
                return None
        return el

    def __getitem__(self, item):
        """Get item function, same for the get[item]"""
        if isinstance(item, tuple):
            return self.get(*item)
        else:
            return self.get(item)

    def __getattr__(self, item):
        """Get attr function, same for the get[item]"""
        return self.get(item)

    def __str__(self):
        """To string function for the whole configuration state"""
        out = ""
        out += "Args:\n" + "\n".join([f"  {str(key)}: {str(value)}" for key, value in self.args.items()]) + "\n"
        out += "Params:\n" + "\n".join([f"  {str(key)}: {str(value)}" for key, value in self.params.items()])
        return out
