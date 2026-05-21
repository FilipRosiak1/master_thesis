import glm

def parseNeuronDToOrient(d: str) -> glm.mat4:
    ix = d.find(':')
    rorient = glm.mat4(1)
    if ix != -1:
        ds = d[ix+1:].split(',')
        ds = [x.split('=') for x in ds]
        rots = [x for x in ds if x[0].startswith('r')]

        if not any("ry" in x[0] for x in rots):
            rots.append(["ry", '0'])
        if not any("rz" in x[0] for x in rots):
            rots.append(["rz", '0'])

        angle = next(float(x[1]) for x in rots if x[0] == "rz")
        rorient = glm.rotate(rorient, angle, glm.vec3(0,0,1))
        angle = -next(float(x[1]) for x in rots if x[0] == "ry")
        rorient = glm.rotate(rorient, angle, glm.vec3(0,1,0))
    return rorient