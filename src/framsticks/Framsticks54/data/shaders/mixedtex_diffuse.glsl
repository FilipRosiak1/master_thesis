glcompat_attrs=5 //FLAG_TEXTURE+FLAG_LIGHT
class_mixedtexture=1

[VERTEX]

#include "common.glsl"

IN mediump vec4 a_vert;
OUT mediump vec4 v_light;
IN vec3 a_norm;
IN vec4 a_texcoord;
OUT mediump vec2 v_texcoord;

void main()
{
gl_Position = u_projmodmat * a_vert;
vec3 ecnormal=normalize(u_rotmat*a_norm);
v_light=max(dot(u_lightdir,ecnormal),0.0)*u_diffuse+u_ambient;
vec3 hv=normalize(-normalize(vec3(u_modmat*a_vert))+u_lightdir);
v_texcoord=vec2(u_texmat*vec4(a_texcoord));
}

[FRAGMENT]

#include "common.glsl"

IN mediump vec4 v_light;
IN mediump vec2 v_texcoord;

void main()
{
lowp vec4 c=u_color;
c*=mix(texture2D(u_tex,v_texcoord),texture2D(u_tex2,v_texcoord),u_mix);
c*=v_light;
gl_FragColor=c;
}
