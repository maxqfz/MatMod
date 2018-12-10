namespace OpenGLProject
{
    public class Shaders
    {
        public static string BkgVertexShader = @"
#version 330

uniform mat4 projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

in vec3 vertexPosition;
in vec2 vertexUV;

out vec2 uv;

void main(void)
{
    uv = vertexUV;
    gl_Position = projection_matrix * view_matrix * model_matrix * vec4(vertexPosition, 1);
}
";

        public static string BkgFragmentShader = @"
#version 330

uniform sampler2D texture;

in vec2 uv;

void main(void)
{
    gl_FragColor = texture2D(texture, uv);
}
";

        public static string CubeVertexShader = @"
#version 330

uniform mat4 projection_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

in vec3 vertexPosition; //Shader position
in vec3 vertexNormal; //Shader normals
in vec3 vertexColor; //Shader color

out vec4 color; //Color output for fragment shader
out vec3 view; //Camera position for fragment shader
out vec3 normal; //Normals output for fragment shader

void main(void)
{
    color = vec4(vertexColor, 1);

    view = view_matrix[3].xyz;
    normal = (model_matrix * vec4(floor(vertexNormal), 0)).xyz;

    gl_Position = projection_matrix * view_matrix * model_matrix * vec4(vertexPosition, 1);
}
";

        public static string CubeFragmentShader = @"
#version 330

vec2 diff = vec2(-326, -46);
int coeff = 628;

uniform sampler2D texture;
//uniform vec3 light_direction;

in vec4 color;
in vec3 view;
in vec3 normal;

void main(void)
{
    //float ambient = 0.3;
    //float diffuse = max(dot(normal, light_direction), 0);
    //float lighting = max(ambient, diffuse);

    vec2 coords = vec2((gl_FragCoord.x + diff.x) / coeff, (gl_FragCoord.y + diff.y) / coeff);
 
	vec3 refr = refract(view, normal, 1.2);

    gl_FragColor = mix(color, texture2D(texture, coords + refr.xy), 0.3); 
}
";
    }
}
