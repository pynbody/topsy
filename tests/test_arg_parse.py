
import topsy

def test_simple_arg_parse():
    args = topsy.parse_args(["test://1000","-q","test-quantity"])
    assert len(args)==1
    args = args[0]
    assert args.filename == "test://1000"
    assert args.quantity == "test-quantity"
    assert args.resolution == topsy.config.DEFAULT_RESOLUTION
    assert args.colormap == topsy.config.DEFAULT_COLORMAP

def test_multi_arg_parse():
    args = topsy.parse_args(["file1","-q","test-quantity","-p","dm","+","file2","-q","test-quantity2"])
    assert len(args)==2

    a = args[0]
    assert a.filename == "file1"
    assert a.quantity == "test-quantity"
    assert a.particle == "dm"

    a = args[1]
    assert a.filename == "file2"
    assert a.quantity == "test-quantity2"
    assert a.particle == "dm"




