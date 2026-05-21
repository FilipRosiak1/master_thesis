glcompat_attrs=260

[VERTEX]
#include "common.glsl"

//regular fixed pipeline for FLAG_TEXTURE+FLAG_HIGHP
IN highp vec4 a_vert;
IN vec4 a_texcoord;
OUT mediump vec2 v_texcoord;
void main()
{
gl_Position = u_projmodmat * a_vert;
v_texcoord=vec2(u_texmat*vec4(a_texcoord));
}

[FRAGMENT]
#include "common.glsl"

IN mediump vec2 v_texcoord;
void main()
{
lowp vec4 c=u_color;
c*=texture2D(u_tex,v_texcoord);
c.a=(c.a*c.a+c.a)*0.5;
gl_FragColor=c;
}
