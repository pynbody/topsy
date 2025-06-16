from topsy.view_synchronizer import ViewSynchronizer

class DummyTarget:
    def __init__(self):
        self.reset_update_count()

    def __setattr__(self, __name, __value):
        super().__setattr__("_num_updates", self._num_updates + 1)
        super().__setattr__(__name, __value)

    def reset_update_count(self):
        super().__setattr__("_num_updates", 0)

    @property
    def update_count(self):
        return self._num_updates

class SubObject:
    def __init__(self):
        self.value = 0

def test_synchronizer():
    """Test the Synchronizer class to ensure it correctly synchronizes attributes between objects."""
    source = DummyTarget()
    target1 = DummyTarget()
    target2 = DummyTarget()

    synchronizer = ViewSynchronizer(['attr1', 'attr2'])
    synchronizer.add_view(source)
    synchronizer.add_view(target1)
    synchronizer.add_view(target2)

    # Set initial values
    source.attr1 = 10
    source.attr2 = 20

    source.reset_update_count()

    synchronizer.perpetuate_update(source)

    assert source.update_count == 0
    assert target1.attr1 == 10
    assert target1.attr2 == 20
    assert target2.attr1 == 10
    assert target2.attr2 == 20

def test_synchronizer_subobjects():
    """Test the Synchronizer class with nested objects."""


    source = DummyTarget()
    source.sub = SubObject()
    target1 = DummyTarget()
    target1.sub = SubObject()
    target2 = DummyTarget()
    target2.sub = SubObject()

    synchronizer = ViewSynchronizer(['sub.value'])
    synchronizer.add_view(source)
    synchronizer.add_view(target1)
    synchronizer.add_view(target2)

    # Set initial values
    source.sub.value = 42

    source.reset_update_count()

    synchronizer.perpetuate_update(source)

    assert source.update_count == 0
    assert target1.sub.value == 42
    assert target2.sub.value == 42

    assert target1.update_count > 0
    assert target2.update_count > 0

    target1.reset_update_count()
    target2.reset_update_count()

    synchronizer.perpetuate_update(target1)
    synchronizer.perpetuate_update(target2)

    # neither of the above should have generated any ops either on targets or source, because
    # they are expected to be "acknowledging receipt" of the update
    assert source.update_count == 0
    assert target1.update_count == 0
    assert target2.update_count == 0

def test_synchronizer_custom_setter():
    """Test the Synchronizer class with a custom setter."""
    class CustomTarget:
        def __init__(self):
            self.result_dict = {}

        def update_value(self, name, value):
            self.result_dict[name] = value

        def get_value(self, name):
            return self.result_dict.get(name, None)

        def __getitem__(self, item):
            return self.get_value(item)

        def __setitem__(self, key, value):
            self.update_value(key, value)




    source = DummyTarget()
    target1 = CustomTarget()

    source.sub = SubObject()
    source.sub.value = 42
    source.value = 1

    synchronizer = ViewSynchronizer(['value', 'sub.value'])
    synchronizer.add_view(source)
    synchronizer.add_view(target1, setter = CustomTarget.update_value, getter = CustomTarget.get_value)

    source.reset_update_count()

    synchronizer.perpetuate_update(source)

    assert source.update_count == 0
    assert not hasattr(target1, 'value')  # Should not have a direct attribute 'value'

    assert target1['value'] == 1
    assert target1['sub.value'] == 42

    synchronizer.perpetuate_update(target1)
    # ^ this is actually just "acknowledging receipt". Otherwise the next update is ignored (part of the infinite
    # loop protection)

    target1['value'] = 2
    target1['sub.value'] = 84

    synchronizer.perpetuate_update(target1)

    assert source.value == 2
    assert source.sub.value == 84


def test_synchronize_with_dict():
    source = DummyTarget()
    source.data = {'key1': 1, 'key2': 2}
    target1 = DummyTarget()
    target1.data = {'key1': 0, 'key2': 0}

    synchronizer = ViewSynchronizer(['data[key1]', 'data[key2]'])
    synchronizer.add_view(source)
    synchronizer.add_view(target1)

    assert target1.data['key1'] == 0
    assert target1.data['key2'] == 0
    synchronizer.perpetuate_update(source)

    assert target1.data['key1'] == 1
    assert target1.data['key2'] == 2
    synchronizer.perpetuate_update(target1)

    assert source.data['key1'] == 1
    assert source.data['key2'] == 2
