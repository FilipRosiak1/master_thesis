import OpenGL.GL as gl
import os
import glm

class ShaderProgram:
    def __init__(self, vertexFile: str, fragmentFile: str, geometryFile: str = None) -> None:
        self.initShaderFromGLSL(vertexFile, fragmentFile, geometryFile)
        self.programID = gl.glCreateProgram()
        gl.glAttachShader(self.programID, self.vs)
        gl.glAttachShader(self.programID, self.fs)
        if geometryFile:
            gl.glAttachShader(self.programID, self.gs)

    def start(self):
        gl.glUseProgram(self.programID)

    def stop(self):
        gl.glUseProgram(0)

    def cleanUp(self):
        self.stop()
        gl.glDetachShader(self.programID, self.vs)
        gl.glDetachShader(self.programID, self.fs)
        gl.glDeleteShader(self.vs)
        gl.glDeleteShader(self.fs)
        gl.glDeleteProgram(self.programID)

    def bindAttributes(self):
        pass

    def getAllUniformLocations(self):
        pass

    def getUniformLocation(self, uniformName):
        return gl.glGetUniformLocation(self.programID, uniformName)

    def bindAttribute(self, attribute: int, variableName: str):
        gl.glBindAttribLocation(self.programID, attribute, variableName)

    def loadFloat(self, location: int, value: float):
        gl.glUniform1f(location, value)

    def loadInt(self, location: int, value: int):
        gl.glUniform1i(location, value)

    def loadVector3(self, location: int, value):
        gl.glUniform3fv(location, 1, glm.value_ptr(value))

    def loadVector2(self, location: int, value):
        gl.glUniform2fv(location, 1, glm.value_ptr(value))

    def loadVector4(self, location: int, value):
        gl.glUniform4fv(location, 1, glm.value_ptr(value))

    def loadBool(self, location: int, value: bool):
        gl.glUniform1f(location, value == 1)

    def loadMatrix4(self, location: int, value):
        gl.glUniformMatrix4fv(location, 1, False, glm.value_ptr(value))

    def initShaderFromGLSL(self, vertex_shader_path, fragment_shader_path, geometry_shader_path = None):
        vertex_shader_source_list = []
        fragment_shader_source_list = []
        geometry_shader_source_list = []
        if(isinstance(vertex_shader_path, str)):
            absDIR =  os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), ".."), vertex_shader_path))
            f = open(absDIR, 'rb')
            vertex_shader_source_list.append(f.read())
            f.close()

            absDIR =  os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), ".."), fragment_shader_path))
            f = open(absDIR, 'rb')
            fragment_shader_source_list.append(f.read())      
            f.close()

            if geometry_shader_path:
                absDIR =  os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), ".."), geometry_shader_path))
                f = open(absDIR, 'rb')
                geometry_shader_source_list.append(f.read())      
                f.close()

            self.initShader(vertex_shader_source_list, fragment_shader_source_list, geometry_shader_source_list)

    def initShader(self, vertex_shader_source_list, fragment_shader_source_list, geometry_shader_source_list):
        # create program
        self.program = gl.glCreateProgram() # pylint: disable=E1111
        #print('create program ',self.program)
        ShaderProgram.printOpenGLError()

        # vertex shader
        #print('compile vertex shader...')
        self.vs = gl.glCreateShader(gl.GL_VERTEX_SHADER) # pylint: disable=E1111
        gl.glShaderSource(self.vs, vertex_shader_source_list)
        gl.glCompileShader(self.vs)
        if gl.GL_TRUE != gl.glGetShaderiv(self.vs, gl.GL_COMPILE_STATUS):
            err = gl.glGetShaderInfoLog(self.vs) 
            raise Exception(err)  
        #gl.glAttachShader(self.program, self.vs)
        ShaderProgram.printOpenGLError()

        # fragment shader
        #print('compile fragment shader...')
        self.fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER) # pylint: disable=E1111
        gl.glShaderSource(self.fs, fragment_shader_source_list)
        gl.glCompileShader(self.fs)
        if gl.GL_TRUE!=gl.glGetShaderiv(self.fs, gl.GL_COMPILE_STATUS):
            err = gl.glGetShaderInfoLog(self.fs) 
            raise Exception(err)       
        #gl.glAttachShader(self.program, self.fs)
        ShaderProgram.printOpenGLError()

        if geometry_shader_source_list:
            self.gs = gl.glCreateShader(gl.GL_GEOMETRY_SHADER) # pylint: disable=E1111
            gl.glShaderSource(self.gs, geometry_shader_source_list)
            gl.glCompileShader(self.gs)
            if gl.GL_TRUE!=gl.glGetShaderiv(self.gs, gl.GL_COMPILE_STATUS):
                err = gl.glGetShaderInfoLog(self.gs) 
                print(err.decode("utf-8"))
                raise Exception(err)       
            #gl.glAttachShader(self.program, self.gs)
            ShaderProgram.printOpenGLError()

        #print('link...')
        #gl.glLinkProgram(self.program)
        #if(gl.GL_TRUE != gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS)):
        #    err =  gl.glGetShaderInfoLog(self.vs) 
        #    raise Exception(err)          
        #printOpenGLError()

    @staticmethod
    def printOpenGLError():
        err = gl.glGetError() # pylint: disable=E1111
        if (err != gl.GL_NO_ERROR):
            print('GLERROR: ', gl.gluErrorString(err)) # pylint: disable=E1101