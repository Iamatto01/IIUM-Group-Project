import java.util.Scanner;

public class  AssignMath {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Select the degree: ");
        int degree = input.nextInt();
        if(degree==1){
            System.out.printf("Enter the value of a: ");
        double a = input.nextDouble();
        System.out.printf("Enter the value of c: ");
        double c = input.nextDouble();

        for (int n = 0; n <= 50; n++) {
            double ans = Math.pow(c, n) * a;
            System.out.println("For n = "+ n +", the answer is: " + ans);
        }
        
        }else if(degree==2){
        System.out.printf("Enter the value of a0: ");
        double[] a = new double[1000];
        a[0] = input.nextDouble();
        
        System.out.printf("Enter the value of a1: ");
        a[1] = input.nextDouble();
        
        System.out.printf("Enter the value of c1: ");
        double c1 = input.nextDouble();
        
        System.out.printf("Enter the value of c2: ");
        double c2 = input.nextDouble();
        double ans=0;
        for(int i=2;i<50;i++){
            a[i] = (double)(c1*a[i-1] + c2*a[i-2]);
        System.out.println(a[i]);
        }
        
        }else{
            System.out.print("Wrong degree!!");
        }
        
        
      input.close();  
    }
}
