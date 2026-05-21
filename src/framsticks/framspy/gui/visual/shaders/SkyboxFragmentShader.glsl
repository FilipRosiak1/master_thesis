#version 330

in vec3 textureCoords;
out vec4 out_Color;

uniform samplerCube modelTexture;

void main(void){
	vec4 textureColor = texture(modelTexture, textureCoords);

    out_Color = textureColor;
	//out_Color = vec4(1, 0, 0, 1);
}