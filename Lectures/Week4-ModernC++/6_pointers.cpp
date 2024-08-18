#include <iostream>

class Motor{
    public:
        int p1, p2, pwm;
        Motor(int p1, int p2, int pwm):
        p1(p1), p2(p2), pwm(pwm){

    }
};

int func(float x){
    return (int)x + 1;
}

int main(){
    using namespace std;

    int i = 100; // 4 bytes
    Motor m(10, 11, 12); // 4 + 4 + 4 bytes

    // taking the address and saving it in a pointer variable
    int     * i_ptr    = & i;   // & -  "Address of" operator
    Motor   * m_ptr    = & m;
    int (*f_ptr)(float)= & func;

    cout << "i          : " << i            << endl;
	cout << "i_ptr      : " << i_ptr        << endl;
	cout << "m.p1       : " << m.p1         << endl; // accesing a member
	cout << "m_ptr      : " << m_ptr        << endl;
	cout << "m_ptr->p1  : " << m_ptr->p1    << endl; // accessning a member form a pointer
	cout << "f_ptr      : " << f_ptr        << endl;
	cout << "*f_ptr(3.0): " << (*f_ptr)(3.0)<< endl;

	int ** i_pp = &i_ptr;  // pointer to a pointer to an int
	cout << "  i_pp: " <<   i_pp << endl;
	cout << " *i_pp: " <<  *i_pp << endl;
	cout << "**i_pp: " << **i_pp << endl;

}