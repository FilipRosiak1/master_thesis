#version 330

in vec2 pass_textureCoordinates;

out vec4 out_Color;

uniform sampler2D modelTexture;
uniform vec3 color;
uniform float textureOn;

void main(void){
	vec4 textureColor = texture(modelTexture, pass_textureCoordinates);
	if(textureColor.a < 0.5){
		discard;
	}
	

	out_Color = textureOn * textureColor + (1.0 - textureOn) * vec4(color, 1);
}