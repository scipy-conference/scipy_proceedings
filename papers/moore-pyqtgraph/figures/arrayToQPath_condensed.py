def arrayToQPath(x, y, connect='all'):
    """Convert an array of x,y coordinats to QPainterPath as efficiently as possible. The *connect* argument
    may be 'all', indicating that each point should be connected to the next; 'pairs', indicating that each
    pair of points should be connected, or an array of int32 values (0 or 1) indicating connections.
    """

    # condensed variant, full code at:
    # https://github.com/pyqtgraph/pyqtgraph/blob/pyqtgraph-0.12.0/pyqtgraph/functions.py#L1558-L1675
    path = QtGui.QPainterPath()
    n = x.shape[0]
    # create empty array, pad with extra space on either end
    # see: https://github.com/qt/qtbase/blob/dev/src/gui/painting/qpainterpath.cpp
    # All values are big endian--pack using struct.pack('>d') or struct.pack('>i')
    arr = np.empty(n+2, dtype=[('c', '>i4'), ('x', '>f8'), ('y', '>f8')])
    # write first two integers
    byteview = arr.view(dtype=np.ubyte)
    byteview[:16] = 0
    byteview.data[16:20] = struct.pack('>i', n)
    arr[1:-1]['x'] = x # Fill array with vertex values
    arr[1:-1]['y'] = y

    # decide which points are connected by lines
    if eq(connect, 'all'):
        arr[1:-1]['c'] = 1
    elif eq(connect, 'pairs'):
        arr[1:-1]['c'][::2] = 0
        arr[1:-1]['c'][1::2] = 1  # connect every 2nd point to every 1st one
    elif eq(connect, 'finite'):
        isfinite = np.isfinite(x) & np.isfinite(y)
        arr[2:]['c'] = isfinite
    elif isinstance(connect, np.ndarray):
        arr[2:-1]['c'] = connect[:-1]
    else:
        raise Exception('connect argument must be "all", "pairs", "finite", or array')
    arr[1]['c'] = 0  # the first vertex has no previous vertex to connect

    byteview.data[-20:-16] = struct.pack('>i', 0)  # cStart
    byteview.data[-16:-12] = struct.pack('>i', 0)  # fillRule (Qt.OddEvenFill)
    
    # create datastream object and stream into path
    path.strn = byteview.data[16:-12]  # make sure data doesn't run away
    buf = QtCore.QByteArray.fromRawData(path.strn)
    ds = QtCore.QDataStream(buf)
    ds >> path
    return path