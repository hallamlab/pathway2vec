import copy


# Arguments for learning models -----------------------------------------------
class Arguments(dict):
    """This class is only used to replace a state variable of Jobman"""

    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(Arguments, self).__getstate__
        elif attr == '__setstate__':
            return super(Arguments, self).__setstate__
        elif attr == '__slots__':
            return super(Arguments, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
        self[attr] = value

    def __str__(self):
        return 'Arguments%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = Arguments()
        for k, kv in self.iteritems():
            z[k] = copy.deepcopy(kv, memo)
        return z
        # -----------------------------------------------------------------------------
