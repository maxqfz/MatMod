using UnityEngine;

public class CameraController : MonoBehaviour
{
    public Transform target;

    public float height;
    public float distance;
    public float speed;

    void FixedUpdate()
    {
        Quaternion rotation = Quaternion.Euler(0, target.eulerAngles.y, 0);

        Vector3 desiredPosition = target.transform.position - rotation * Vector3.forward * distance;
        desiredPosition.y = target.position.y + height;

        Vector3 smoothedPosition = Vector3.Lerp(transform.position, desiredPosition, speed);

        transform.position = smoothedPosition;
        transform.LookAt(target.transform);
    }
}