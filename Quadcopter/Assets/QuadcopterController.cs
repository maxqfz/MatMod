using UnityEngine;

public class QuadcopterController : MonoBehaviour
{
    /* GAME OBJECTS */
    //Квадрик
    private Rigidbody Quadcopter;
    //Винты квадрика
    public GameObject PropellerFL;
    public GameObject PropellerFR;
    public GameObject PropellerBL;
    public GameObject PropellerBR;

    /* QUADCOPTER PARAMS */
    [Header("Quadcopter Parameters")]
    public float maxPropellerForce; //5
    public float maxTorque; //5
    public float maxThrottle; //5

    //Movement factors
    public bool useJoystick; //Использовать джойстик для управления?
    public float pitch; //Тангаж (движение вперёд-назад)
    public float roll; //Крен (движение влево-вправо)
    public float yaw; //Рыскание (поворот)
    public float throttle; //Газ

    /* WIND */
    [Header("Wind")]
    public float windForce; //Сила ветра
    public float windDirection; //Направление ветра
    
    /* PID controllers */
    private PIDController PIDPitch = new PIDController(0.01F, 0.001F, 0.005F);
    private PIDController PIDRoll = new PIDController(0.01F, 0.001F, 0.005F);
    private PIDController PIDYaw = new PIDController(0.5F, 0, 0.005F);

    void Start()
    {
        Quadcopter = gameObject.GetComponent<Rigidbody>();
    }

    void Update()
    {
        /* Считывание управление */
        const float pitchMax = 45F;
        const float rollMax = 45F;
        const float yawVelocity = 3F;
        const float throttleStep = 0.1F;
        
        //Проверяем, что подключен джойстик
        if(useJoystick && Input.GetJoystickNames().Length > 0)
        {
            //Тангаж
            pitch = pitchMax * -Input.GetAxis("JoystickY");
            //Крен
            roll = rollMax * -Input.GetAxis("JoystickX");
            //Рыскание
            yaw = yawVelocity * Input.GetAxis("JoystickZ");
            if (Input.GetKey(KeyCode.JoystickButton3))
                yaw = -yawVelocity;
            if (Input.GetKey(KeyCode.JoystickButton1))
                yaw = +yawVelocity;
            //Газ
            throttle += throttleStep * -Input.GetAxis("JoystickT");
            if (Input.GetKey(KeyCode.JoystickButton0))
                throttle += throttleStep;
            if (Input.GetKey(KeyCode.JoystickButton2))
                throttle -= throttleStep;
        }
        else
        {
            //Тангаж
            pitch = 0;
            if (Input.GetKey(KeyCode.W))
                pitch = pitchMax;
            else if (Input.GetKey(KeyCode.S))
                pitch = -pitchMax;
            //Крен
            roll = 0;
            if (Input.GetKey(KeyCode.A))
                roll = rollMax;
            else if (Input.GetKey(KeyCode.D))
                roll = -rollMax;
            //Рыскание
            yaw = 0;
            if (Input.GetKey(KeyCode.K))
                yaw -= yawVelocity;
            if (Input.GetKey(KeyCode.Semicolon))
                yaw += yawVelocity;
            //Газ
            if (Input.GetKey(KeyCode.O))
                throttle += throttleStep;
            if (Input.GetKey(KeyCode.L))
                throttle -= throttleStep;
        }
        throttle = Mathf.Clamp(throttle, 0f, maxThrottle);
    }

    void FixedUpdate()
    {
        Stabilize();
        AddWind();
    }

    void Stabilize()
    {
        //Получаем текущий поворот квадрика по X и Z
        float pitchCurrent = NormalizeAngle180(transform.eulerAngles.x);
        float rollCurrent = NormalizeAngle180(transform.eulerAngles.z);

        //Получаем ошибку (разницу между текущим и желаемым)
        float pitchError = pitchCurrent - pitch;
        float rollError = rollCurrent - roll;

        //Получаем необходимую коррекцию
        float pitchCorrection = PIDPitch.Calculate(pitchError);
        float rollCorrection = PIDRoll.Calculate(rollError);

        //Считаем силу для винтов
        float propellerForceFL = throttle + pitchCorrection + rollCorrection;
        float propellerForceFR = throttle + pitchCorrection - rollCorrection;
        float propellerForceBL = throttle - pitchCorrection + rollCorrection;
        float propellerForceBR = throttle - pitchCorrection - rollCorrection;

        //Применяем силу к винтам (не больше макс.)
        AddForceToPropeller(PropellerFL, Mathf.Clamp(propellerForceFL, 0, maxPropellerForce));
        AddForceToPropeller(PropellerFR, Mathf.Clamp(propellerForceFR, 0, maxPropellerForce));
        AddForceToPropeller(PropellerBL, Mathf.Clamp(propellerForceBL, 0, maxPropellerForce));
        AddForceToPropeller(PropellerBR, Mathf.Clamp(propellerForceBR, 0, maxPropellerForce));

        //Получаем текущее вращение по Y
        float yawCurrent = Quadcopter.angularVelocity.y;

        //Считаем ошибку
        float yawError = yawCurrent - yaw;

        //Получаем коррекцию
        float yawCorrection = PIDYaw.Calculate(yawError);

        /*
        //Считаем крутящий момент для винтов
        float propellerTorqueFL = -yawCorrection;// throttle - yawCorrection;
        float propellerTorqueFR = -yawCorrection;// -throttle - yawCorrection;
        float propellerTorqueBL = -yawCorrection;// -throttle - yawCorrection;
        float propellerTorqueBR = -yawCorrection;// throttle - yawCorrection;

        //Применяем крутящий момент к винтам
        AddTorqueToPropeller(PropellerFL, Mathf.Clamp(propellerTorqueFL, -maxTorque, maxTorque));
        AddTorqueToPropeller(PropellerFR, Mathf.Clamp(propellerTorqueFR, -maxTorque, maxTorque));
        AddTorqueToPropeller(PropellerBL, Mathf.Clamp(propellerTorqueBL, -maxTorque, maxTorque));
        AddTorqueToPropeller(PropellerBR, Mathf.Clamp(propellerTorqueBR, -maxTorque, maxTorque));
        */

        //Добавляем вращение к квадрику
        Quadcopter.AddTorque(transform.up * Mathf.Clamp(-yawCorrection, -maxTorque, maxTorque));
    }

    //Добавляем силу к винту
    void AddForceToPropeller(GameObject propeller, float force)
    {
        propeller.GetComponent<Rigidbody>().AddForce(propeller.transform.up * force);
    }

    //Добавляем крутящий момент к винту
    void AddTorqueToPropeller(GameObject propeller, float torque)
    {
        propeller.GetComponent<Rigidbody>().AddTorque(propeller.transform.up * torque);
    }

    //Нормализуем угол до промежутка [-180, 180)
    private float NormalizeAngle180(float angle)
    {
        if (angle >= 180)
            return angle - 360F;
        return angle;
    }

    //Добавляем внешние силы (напр. ветер)
    private void AddWind()
    {
        Vector3 wind = Quaternion.Euler(0, windDirection, 0) * (-Vector3.forward);
        Quadcopter.AddForce(wind * windForce);
    }
}

public class PIDController
{
    //Параметры PID
    readonly float Kp;
    readonly float Ki;
    readonly float Kd;

    private float sumError = 0F; //Сумма ошибок
    private float sumErrorMax = 1F; //Максимальное значение суммы ошибок
    private float prevError = 0F; //Предыдущая ошибка

    public PIDController(float Kp, float Ki, float Kd)
    {
        this.Kp = Kp;
        this.Ki = Ki;
        this.Kd = Kd;
    }

    public float Calculate(float error)
    {
        //Proportional
        float P = Kp * error;

        //Integral
        sumError += error;
        float I = Ki * (Mathf.Clamp(sumError, -sumErrorMax, sumErrorMax) * Time.fixedDeltaTime);

        //Derivative
        float D = Kd * ((error - prevError) / Time.fixedDeltaTime);
        prevError = error;

        return P + I + D;
    }
}