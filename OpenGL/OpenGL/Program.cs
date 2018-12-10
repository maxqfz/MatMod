using System;
using System.Linq;
using OpenGL;
using Tao.FreeGlut;

namespace OpenGLProject
{
    class Program
    {
        const int WINDOW_WIDTH = 1280, WINDOW_HEIGHT = 720;
        static private ShaderProgram cubeProgram, bkgProgram;
        //Texture
        private static Texture texture;
        //Cube
        private static VBO<Vector3> cube, cubeColor;
        private static VBO<int> cubeQuads;
        private static VBO<Vector3> cubeNormal;
        //Cover
        private static VBO<Vector3> bkg;
        private static VBO<Vector2> bkgCoords;
        private static VBO<int> bkgQuads;
        //Rotation angles
        static float xangle, yangle;
        //Rotation step
        const float step = 0.02F;
        //Timer for rotation
        private static System.Diagnostics.Stopwatch stopwatch;
        //Glut params
        private static bool fullscreen;

        static void Main(string[] args)
        {
            //Create an OpenGL window
            Glut.glutInit();
            //Double-buffering to avoid flickering while rotating and depth-buffering
            //Glut.glutInitDisplayMode(Glut.GLUT_DOUBLE | Glut.GLUT_DEPTH);
            //Set UI window size
            Glut.glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
            Glut.glutCreateWindow("OpenGLProject");

            //Enable depth testing to ensure correct z-ordering of our fragments
            Gl.Enable(EnableCap.DepthTest);
            //Enable blend to be able to add transparent objects 
            Gl.Enable(EnableCap.Blend);
            Gl.BlendFunc(BlendingFactorSrc.SrcAlpha, BlendingFactorDest.OneMinusSrcAlpha);

            //Provide necessary Glut callbacks
            //Glut.glutDisplayFunc(OnRender);
            Glut.glutIdleFunc(OnRender);
            Glut.glutKeyboardFunc(OnKeyDown);
            Glut.glutCloseFunc(OnClose);

            //Compile our shaders
            cubeProgram = new ShaderProgram(Shaders.CubeVertexShader, Shaders.CubeFragmentShader);
            bkgProgram = new ShaderProgram(Shaders.BkgVertexShader, Shaders.BkgFragmentShader);
            Matrix4 view_matrix = Matrix4.LookAt(new Vector3(0, 0, 10), Vector3.Zero, new Vector3(0, 1, 0));
            Matrix4 projection_matrix = Matrix4.CreatePerspectiveFieldOfView(0.45f, (float)WINDOW_WIDTH / WINDOW_HEIGHT, 0.1f, 1000f);

            cubeProgram.Use();
            cubeProgram["view_matrix"].SetValue(view_matrix);
            cubeProgram["projection_matrix"].SetValue(projection_matrix);
            //cubeProgram["light_direction"].SetValue(new Vector3(0, 0, 1));

            //Create a cube
            cube = new VBO<Vector3>(new Vector3[] {
                new Vector3(1, 1, -1), new Vector3(-1, 1, -1), new Vector3(-1, 1, 1), new Vector3(1, 1, 1), //TOP
                new Vector3(1, -1, 1), new Vector3(-1, -1, 1), new Vector3(-1, -1, -1), new Vector3(1, -1, -1), //BOTTOM
                new Vector3(1, 1, 1), new Vector3(-1, 1, 1), new Vector3(-1, -1, 1), new Vector3(1, -1, 1), //FRONT
                new Vector3(1, -1, -1), new Vector3(-1, -1, -1), new Vector3(-1, 1, -1), new Vector3(1, 1, -1), //BACK
                new Vector3(-1, 1, 1), new Vector3(-1, 1, -1), new Vector3(-1, -1, -1), new Vector3(-1, -1, 1), //LEFT
                new Vector3(1, 1, -1), new Vector3(1, 1, 1), new Vector3(1, -1, 1), new Vector3(1, -1, -1), //RIGHT
            });
            cubeNormal = new VBO<Vector3>(new Vector3[] {
                new Vector3(0, 1, 0), new Vector3(0, 1, 0), new Vector3(0, 1, 0), new Vector3(0, 1, 0), //TOP NORMAL
                new Vector3(0, -1, 0), new Vector3(0, -1, 0), new Vector3(0, -1, 0), new Vector3(0, -1, 0), //BOTTOM NORMAL
                new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1), //FRONT NORMAL
                new Vector3(0, 0, -1), new Vector3(0, 0, -1), new Vector3(0, 0, -1), new Vector3(0, 0, -1), //BACK NORMAL
                new Vector3(-1, 0, 0), new Vector3(-1, 0, 0), new Vector3(-1, 0, 0), new Vector3(-1, 0, 0), //LEFT NORMAL
                new Vector3(1, 0, 0), new Vector3(1, 0, 0), new Vector3(1, 0, 0), new Vector3(1, 0, 0) }); //RIGHT NORMAL
            cubeColor = new VBO<Vector3>(new Vector3[] {
                new Vector3(0, 1, 0), new Vector3(0, 1, 0), new Vector3(0, 1, 0), new Vector3(0, 1, 0), //GREEN
                new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1), new Vector3(0, 0, 1), //BLUE
                new Vector3(1, 1, 1), new Vector3(1, 1, 1), new Vector3(1, 1, 1), new Vector3(1, 1, 1), //WHITE
                new Vector3(1, 0, 0), new Vector3(1, 0, 0), new Vector3(1, 0, 0), new Vector3(1, 0, 0), //RED
                new Vector3(1, 0, 1), new Vector3(1, 0, 1), new Vector3(1, 0, 1), new Vector3(1, 0, 1), //PURPLE
                new Vector3(1, 1, 0), new Vector3(1, 1, 0), new Vector3(1, 1, 0), new Vector3(1, 1, 0), //YELLOW
            });
            cubeQuads = new VBO<int>(Enumerable.Range(0, 24).ToArray(), BufferTarget.ElementArrayBuffer);

            bkgProgram.Use();
            bkgProgram["view_matrix"].SetValue(view_matrix);
            bkgProgram["projection_matrix"].SetValue(projection_matrix);

            //Create a square cover
            bkg = new VBO<Vector3>(new Vector3[] {
                new Vector3(10, 10, 10), new Vector3(-10, 10, 10), new Vector3(-10, -10, 10), new Vector3(10, -10, 10), });
            bkgCoords = new VBO<Vector2>(new Vector2[] {
                new Vector2(1, 1), new Vector2(0, 1), new Vector2(0, 0), new Vector2(1, 0), });
            bkgQuads = new VBO<int>(Enumerable.Range(0, 4).ToArray(), BufferTarget.ElementArrayBuffer);
            texture = new Texture("texture.jpg");

            Glut.glutMainLoop();
        }

        private static void OnRender()
        {
            //Set up the OpenGL viewport
            Gl.Viewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
            //Clear both the color and depth bits
            Gl.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

            BkgRender();
            CubeRender();

            //Flush buffer to display
            Gl.Flush();
            //Glut.glutSwapBuffers();
        }

        private static void BkgRender()
        {
            bkgProgram.Use();
            bkgProgram["model_matrix"].SetValue(Matrix4.CreateTranslation(new Vector3(0, 0, -50)));

            Gl.BindBufferToShaderAttribute(bkg, bkgProgram, "vertexPosition");
            Gl.BindBufferToShaderAttribute(bkgCoords, bkgProgram, "vertexUV");
            Gl.BindBuffer(bkgQuads);
            Gl.BindTexture(texture);

            //Draw background as quads
            Gl.DrawElements(BeginMode.Quads, bkgQuads.Count, DrawElementsType.UnsignedInt, IntPtr.Zero);
        }

        private static void CubeRender()
        {
            if (stopwatch != null)
            {
                xangle += (float)stopwatch.ElapsedMilliseconds / System.Diagnostics.Stopwatch.Frequency;
                yangle += (float)stopwatch.ElapsedMilliseconds / System.Diagnostics.Stopwatch.Frequency / 2;
            }

            cubeProgram.Use();
            //Rotate and translate the square
            cubeProgram["model_matrix"].SetValue(Matrix4.CreateRotationX(xangle) * Matrix4.CreateRotationY(yangle) * Matrix4.CreateTranslation(new Vector3(0, 0, 0)));
            
            //Bind the vertex attribute arrays for the square
            Gl.BindBufferToShaderAttribute(cube, cubeProgram, "vertexPosition");
            Gl.BindBufferToShaderAttribute(cubeColor, cubeProgram, "vertexColor");
            Gl.BindBufferToShaderAttribute(cubeNormal, cubeProgram, "vertexNormal");
            Gl.BindBuffer(cubeQuads);
            Gl.BindTexture(texture);

            //Draw the cube as quads
            Gl.DrawElements(BeginMode.Quads, cubeQuads.Count, DrawElementsType.UnsignedInt, IntPtr.Zero);
        }

        private static void OnKeyDown(byte key, int x, int y)
        {
            if (key == 'w') xangle -= step;
            else if (key == 's') xangle += step;
            else if (key == 'a') yangle -= step;
            else if (key == 'd') yangle += step;
            else if (key == ' ') stopwatch = stopwatch == null ? System.Diagnostics.Stopwatch.StartNew() : null;
            else if (key == 'f')
            {
                fullscreen = !fullscreen;
                if (fullscreen)
                    Glut.glutFullScreen();
                else
                    Glut.glutPositionWindow(0, 0);
            }
            else if (key == 27) Glut.glutLeaveMainLoop();
        }

        private static void OnClose()
        {
            //TODO: Dispose all objects created
            cubeProgram.DisposeChildren = true;
            cubeProgram.Dispose();
        }
    }
}
