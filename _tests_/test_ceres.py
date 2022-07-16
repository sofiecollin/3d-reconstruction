from build.ceres_python import add, Pet

# A test file to integrate Ceres to Python

def test_add():
    print(add(8, 3))

def test_pet():
    my_dog = Pet('Pluto', 5, 'Lennart')
    assert my_dog.get_name() == 'Pluto'
    assert my_dog.get_hunger() == 5
    my_dog.go_for_a_walk()
    assert my_dog.get_hunger() == 6
    print(my_dog.get_name())
    print(my_dog.get_owner())

def test_ceres(R,t):
    test_add()
    test_pet()