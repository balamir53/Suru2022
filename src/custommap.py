# https://molotovcockatiel.com/hex-map-maker/
#in this site, 1: "Clear", 2: "Light Brush", 3: "Forest", 4: "Urban"
# terrain types = 1: "g", 2: "d", 3: "w", 4: "m"
# keys are self.height of maps
maps = {
    "18":[
    [[1,1,4,4,1,1,3,2,1,1,1,1],[1,2,4,1,1,1,3,1,1,1,1,1],[1,1,4,4,1,1,3,2,1,3,4,1],[1,2,4,1,1,1,3,1,1,4,2,1],[1,1,4,4,1,1,3,2,1,3,4,1],[1,2,4,1,1,3,3,1,1,4,2,1],[1,1,4,3,1,1,3,2,1,3,4,1],[1,2,4,3,1,1,3,2,1,3,2,1],[1,1,2,3,3,1,1,2,1,1,2,1],[1,1,3,3,1,1,2,1,1,1,1,1],[1,1,1,3,1,1,1,2,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,2,1,1],[1,1,1,1,1,1,1,1,2,2,1,1],[1,1,4,1,1,4,1,1,4,4,1,1],[1,4,1,1,1,4,1,1,4,2,1,1],[1,1,4,1,1,4,1,1,4,2,1,1],[1,4,1,1,4,4,1,1,4,2,1,1],[1,1,1,1,1,4,4,1,4,4,1,1],[1,1,1,1,4,4,1,1,4,1,1,1],[1,1,1,1,3,4,4,1,1,4,1,1],[1,1,1,1,3,3,1,1,4,1,1,1],[1,1,1,1,1,3,1,1,1,1,1,1],[1,1,1,1,1,3,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,3,3,1,1,1,1],[1,1,1,3,1,1,3,3,1,1,1,1],[1,1,3,3,1,1,4,4,1,1,1,1],[1,1,4,3,1,1,3,4,4,1,1,1],[1,4,4,3,1,1,4,4,1,1,1,1],[1,1,4,3,3,1,1,4,4,1,1,1],[1,4,4,3,1,1,1,4,1,1,1,1]],
    [[1,1,4,4,4,4,4,4,4,4,4,1],[1,4,4,4,4,4,4,4,4,4,1,1],[1,1,4,4,4,4,4,4,4,4,1,1],[1,1,4,4,4,4,4,4,4,4,1,1],[1,1,1,4,4,4,4,4,4,4,1,1],[1,1,1,4,4,4,4,4,4,1,1,1],[1,1,1,1,4,4,4,4,4,1,1,1],[1,1,1,1,4,4,4,4,1,1,1,1],[1,1,1,1,1,4,4,4,1,1,1,1],[1,1,1,1,1,4,4,4,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,2,2,1,1,1,1,1],[1,4,1,1,2,2,2,1,1,4,1,1],[1,1,4,1,2,3,3,2,1,4,1,1],[1,4,4,1,3,3,3,1,4,4,1,1],[1,1,4,1,2,3,3,2,1,4,4,1],[1,4,4,1,3,3,3,1,4,4,1,1],[1,1,4,1,2,3,3,2,1,4,4,1],[1,1,1,1,3,3,3,1,1,4,1,1],[1,1,1,1,2,3,3,2,1,1,1,1],[1,1,1,1,3,3,3,1,1,1,1,1],[1,1,1,1,2,3,3,2,1,1,1,1],[1,1,1,1,2,2,2,1,1,1,1,1],[1,1,1,1,1,2,2,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,4,1,1,1,1,1,1,1],[1,1,1,4,4,4,1,4,4,1,1,1],[1,1,1,4,4,4,4,4,4,4,1,1],[1,1,4,4,4,4,4,4,4,4,1,1],[1,1,4,4,4,4,4,4,4,4,4,1],[1,4,4,4,4,4,4,4,4,4,1,1]],
    [[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,2,2,1,1],[1,2,2,1,1,1,1,2,2,2,1,1],[1,1,3,2,1,1,1,1,3,3,2,1],[1,2,3,2,1,1,1,2,3,3,1,1],[1,1,3,3,1,1,1,1,3,3,2,1],[1,2,3,2,1,1,1,2,3,3,1,1],[1,1,3,3,1,1,1,1,2,2,2,1],[1,2,3,2,1,1,1,1,2,2,1,1],[1,1,2,3,1,1,1,1,1,1,1,1],[1,1,2,2,1,4,1,1,1,1,1,1],[1,1,1,2,1,4,4,1,1,1,1,1],[1,1,1,1,4,4,4,1,1,1,1,1],[1,1,1,1,4,4,4,4,1,1,1,1],[1,1,1,1,4,4,4,1,1,1,1,1],[1,1,1,1,4,4,4,4,1,1,1,1],[1,1,1,1,4,4,4,4,1,1,1,1],[1,1,1,1,4,4,4,4,1,2,1,1],[1,1,2,1,4,4,4,1,2,2,1,1],[1,1,2,2,1,4,4,1,1,3,2,1],[1,2,3,2,4,4,4,1,2,3,1,1],[1,1,3,3,1,4,4,1,1,3,3,1],[1,2,3,2,1,1,1,1,2,3,1,1],[1,1,3,3,1,1,1,1,2,3,3,1],[1,2,3,2,1,1,1,1,2,3,1,1],[1,1,3,2,1,1,1,1,2,2,2,1],[1,2,2,1,1,1,1,1,2,2,1,1],[1,1,2,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1]],
    [[1,1,1,4,4,4,1,1,1,1,1,1],[1,1,4,4,4,4,1,1,1,1,1,1],[1,1,1,4,4,4,1,1,1,1,1,1],[1,1,1,4,4,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,4,1,1,1],[1,1,2,1,1,1,1,4,4,1,1,1],[1,1,2,1,1,1,1,1,4,4,1,1],[1,1,2,1,1,1,1,1,4,1,1,1],[1,1,2,2,1,1,1,1,1,1,1,1],[1,1,3,1,1,1,1,1,1,1,1,1],[1,1,3,3,1,4,1,1,1,1,1,1],[1,1,3,1,1,4,1,1,2,1,1,1],[1,1,3,3,1,4,4,1,2,2,1,1],[1,1,3,1,1,4,4,1,2,1,1,1],[1,1,2,2,1,4,4,1,2,2,1,1],[1,1,2,1,1,1,1,1,2,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,4,1,1],[1,1,1,1,1,1,1,1,1,4,1,1],[1,1,1,1,1,1,3,1,1,4,1,1],[1,1,1,1,1,1,3,3,1,4,1,1],[1,1,4,1,1,3,3,1,2,4,1,1],[1,1,2,4,1,2,3,1,1,4,1,1],[1,1,4,4,1,2,2,1,2,1,1,1],[1,1,2,4,1,1,2,1,1,1,1,1],[1,1,4,4,1,1,2,1,1,1,1,1],[1,1,2,2,4,1,1,1,1,1,1,1],[1,1,2,2,1,2,1,1,1,1,1,1],[1,1,1,1,1,1,4,1,1,1,1,1],[1,1,1,1,1,2,4,1,1,1,1,1],[1,1,1,1,1,2,4,4,1,1,1,1],[1,1,1,1,1,4,4,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1]],
    [[1,1,4,4,4,1,3,1,1,1,1,1],[1,1,4,4,1,3,3,1,1,1,1,1],[1,1,4,4,4,1,3,1,1,1,1,1],[1,1,4,4,1,1,3,1,1,1,1,1],[1,1,1,4,1,1,3,3,1,1,1,1],[1,1,4,1,1,1,3,3,1,1,1,1],[1,1,1,1,1,1,1,3,3,1,1,1],[1,1,1,1,1,1,1,3,1,1,1,1],[1,1,1,1,1,1,1,3,3,1,1,1],[1,1,1,1,1,1,1,3,1,1,1,1],[1,1,1,1,1,1,1,1,3,1,1,1],[1,1,1,2,1,1,1,3,3,1,1,1],[1,1,1,2,1,1,1,1,3,1,1,1],[1,1,2,2,1,3,1,1,3,1,1,1],[1,1,1,2,1,3,3,1,3,1,1,1],[1,1,2,2,1,3,1,1,3,1,1,1],[1,1,1,2,1,3,1,1,3,1,1,1],[1,1,2,1,1,3,1,1,3,1,1,1],[1,1,1,1,1,3,1,1,3,1,1,1],[1,1,1,1,3,3,1,1,3,1,1,1],[1,1,1,1,1,3,1,1,3,1,1,1],[1,1,1,1,3,1,1,3,3,2,1,1],[1,1,1,1,3,3,1,1,3,1,2,1],[1,1,1,3,3,1,1,3,1,2,1,1],[1,1,1,3,3,1,1,1,3,1,2,1],[1,1,3,3,1,1,1,1,1,2,1,1],[1,1,3,3,1,1,1,1,1,1,2,1],[1,3,3,1,1,1,1,1,1,2,1,1],[1,1,3,1,1,1,1,1,1,1,1,1],[1,3,1,1,1,1,4,1,1,2,1,1],[1,1,1,1,1,1,4,4,1,1,1,1],[1,1,1,1,1,1,4,4,1,1,1,1],[1,1,1,1,1,1,4,4,1,1,1,1],[1,1,1,1,1,1,4,4,1,1,1,1],[1,1,1,1,1,1,4,4,4,1,1,1],[1,1,1,1,1,1,4,4,1,1,1,1]],
    [[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,3,1,1,1,1,1,1,1],[1,1,1,3,1,1,1,1,1,1,1,1],[1,1,1,3,3,1,1,1,1,1,1,1],[1,1,3,3,1,1,1,1,1,1,1,1],[1,1,1,3,3,1,1,1,4,1,1,1],[1,1,3,3,1,2,1,4,4,1,1,1],[1,1,1,3,3,2,1,1,4,1,1,1],[1,1,2,3,3,2,1,4,4,1,1,1],[1,1,1,2,2,2,1,1,4,1,1,1],[1,1,1,2,2,1,1,4,4,1,1,1],[1,1,1,1,1,2,1,2,4,1,1,1],[1,1,1,1,1,1,1,4,4,1,1,1],[1,1,1,1,1,1,1,2,4,1,1,1],[1,1,1,1,1,1,1,2,2,1,1,1],[1,1,2,1,1,1,1,2,2,1,1,1],[1,1,2,2,1,1,1,2,1,1,1,1],[1,1,2,2,1,1,1,1,1,1,1,1],[1,1,4,4,1,1,1,1,1,1,1,1],[1,1,4,4,4,1,1,1,1,1,1,1],[1,1,4,4,1,1,1,1,1,1,1,1],[1,1,4,4,4,1,1,2,1,1,1,1],[1,1,2,4,1,1,2,2,1,1,1,1],[1,1,2,4,1,1,1,2,2,1,1,1],[1,1,2,4,1,1,3,2,1,1,1,1],[1,1,1,4,1,1,3,3,2,1,1,1],[1,1,2,4,1,3,3,2,1,1,1,1],[1,1,1,4,1,1,3,3,1,1,1,1],[1,1,2,4,1,3,3,3,1,1,1,1],[1,1,1,1,1,1,3,3,1,1,1,1],[1,1,1,1,1,3,3,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1,1,1,1]],
    # [],
    # [],
    # [],
    # [],
    # [],
    # [],
    # [],
    # [],
    ],
    "8":[
    [[1,1,3,3,1,1],[1,2,3,1,1,1],[1,1,2,3,1,1],[1,2,3,3,1,1],[1,1,3,3,1,1],[1,1,3,3,1,1],[1,1,1,3,1,1],[1,2,1,1,4,1],[1,4,1,1,2,2],[1,2,1,1,2,1],[1,2,1,1,1,1],[1,1,4,1,1,1],[1,1,1,4,1,1],[1,1,4,2,1,1],[1,1,4,4,2,1],[1,4,4,4,1,1]],
    [[1,1,2,1,3,1],[1,2,2,3,3,1],[1,1,1,1,3,1],[1,4,1,3,3,1],[1,2,4,1,3,1],[1,4,1,2,1,1],[1,2,1,1,1,1],[1,2,1,2,1,1],[1,1,1,1,4,1],[1,1,1,2,1,1],[1,1,3,1,4,1],[1,3,3,2,4,1],[1,1,3,1,1,1],[1,3,3,1,1,1],[1,1,3,1,1,1],[1,1,1,1,1,1]],
    [[1,1,2,1,3,1],[1,2,2,3,3,1],[1,1,1,1,3,1],[1,4,1,3,3,1],[1,2,4,1,3,1],[1,4,1,2,1,1],[1,2,1,1,1,1],[1,2,1,2,1,1],[1,1,1,1,4,1],[1,1,1,2,1,1],[1,1,3,1,4,1],[1,3,3,2,4,1],[1,1,3,1,1,1],[1,3,3,1,1,1],[1,1,3,1,1,1],[1,1,1,1,1,1]],
    [[1,1,2,1,3,1],[1,2,2,3,3,1],[1,1,1,1,3,1],[1,4,1,3,3,1],[1,2,4,1,3,1],[1,4,1,2,1,1],[1,2,1,1,1,1],[1,2,1,2,1,1],[1,1,1,1,4,1],[1,1,1,2,1,1],[1,1,3,1,4,1],[1,3,3,2,4,1],[1,1,3,1,1,1],[1,3,3,1,1,1],[1,1,3,1,1,1],[1,1,1,1,1,1]],
    [[1,1,2,1,3,1],[1,2,2,3,3,1],[1,1,1,1,3,1],[1,4,1,3,3,1],[1,2,4,1,3,1],[1,4,1,2,1,1],[1,2,1,1,1,1],[1,2,1,2,1,1],[1,1,1,1,4,1],[1,1,1,2,1,1],[1,1,3,1,4,1],[1,3,3,2,4,1],[1,1,3,1,1,1],[1,3,3,1,1,1],[1,1,3,1,1,1],[1,1,1,1,1,1]],
    [[1,1,2,1,3,1],[1,2,2,3,3,1],[1,1,1,1,3,1],[1,4,1,3,3,1],[1,2,4,1,3,1],[1,4,1,2,1,1],[1,2,1,1,1,1],[1,2,1,2,1,1],[1,1,1,1,4,1],[1,1,1,2,1,1],[1,1,3,1,4,1],[1,3,3,2,4,1],[1,1,3,1,1,1],[1,3,3,1,1,1],[1,1,3,1,1,1],[1,1,1,1,1,1]],
    [[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]],
    [[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1]],
    [[1,4,1,1,4,1],[2,3,1,2,3,1],[1,4,1,1,4,1],[2,3,1,2,3,1],[1,2,1,1,4,1],[2,3,1,2,3,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,1,1,1,1,1],[1,3,2,1,4,1],[1,2,1,2,3,1],[1,3,2,1,4,1],[1,4,1,2,3,1],[1,3,2,1,4,1],[1,4,1,2,3,1]]
    ]}