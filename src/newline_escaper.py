
samples = [
    """Random rn;
int n = maximum - minimum + 1;
int i = <MARK>""","""void f(String[] first, String[] second) {
    String[] both = <MARK>
}""","""public static void main(String[] args) {
    Collection<Integer> l = new ArrayList<>();

    for (int i = 0; i < 10; ++i) {
        l.add(4);
        l.add(5);
        l.add(6);
    }

    <MARK>

    System.out.println(l);
}""","""public static String getCurrentTimeStamp() {
    SimpleDateFormat sdfDate = new SimpleDateFormat(<MARK>);//dd/MM/yyyy
    Date now = new Date();
    String strDate = sdfDate.format(now);
    return strDate;
}""","""public class ApplesDO {

    private String apple;

    public String getApple() {
        return apple;
    }

    public void setApple(String appl) {
        this.apple = apple;
    }

    public ApplesDO(CustomType custom){
        //constructor Code
    }
    <MARK>
}
""","""List<String> supplierNames = new <MARK>;
supplierNames.add("sup1");
supplierNames.add("sup2");
supplierNames.add("sup3");
System.out.println(supplierNames.get(1));""","""from functools import reduce
l = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
flat_list = <MARK>
""","""items = []
items.append("apple")
items.append("orange")
items.append("banana")
<MARK>""","""x = " <MARK> {0} "
print(x.format(42))""","""from pylab import figure, axes, pie, title, show

# Make a square figure and axes
figure(1, figsize=(6, 6))
ax = axes([0.1, 0.1, 0.8, 0.8])

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
fracs = [15, 30, 45, 10]

explode = (0, 0.05, 0, 0)
pie(fracs, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)
title('Raining Hogs and Dogs', bbox={'facecolor': '0.8', 'pad': 5})

<MARK>""","""agent_telno = agent.find('div', 'agent_contact_number')
agent_telno = '' if agent_telno is None else agent_telno.contents[0]
s = agent_contact + u' ' + agent_telno
p.agent_info = <MARK>.strip()""","""obj = open('data.txt', 'wb')
<MARK>
obj.close""", """@Service
<MARK>
public class MileageFeeCalculator {

    @Autowired
    private MileageRateService rateService; // <--- should be autowired, is null

    public float mileageCharge(final int miles) {
        return (miles * rateService.ratePerMile()); // <--- throws NPE
    }
}""","""String decompressGZIP(byte[] gzip) throws IOException {
    java.util.zip.Inflater inf = new java.util.zip.Inflater();
    java.io.ByteArrayInputStream bytein = new java.io.ByteArrayInputStream(gzip);
    java.util.zip.GZIPInputStream gzin = new java.util.zip.GZIPInputStream(bytein);
    java.io.ByteArrayOutputStream byteout = new java.io.ByteArrayOutputStream();
    int res = 0;
    byte buf[] = new byte[1024];
    while (res >= 0) {
        res = gzin.read(buf, 0, buf.length);
        if (res > 0) {
            byteout.write(buf, 0, res);
        }
    }
    byte uncompressed[] = byteout.toByteArray();
    return <MARK>;
}""","""BigDecimal a = new BigDecimal("1.6");
BigDecimal b = new BigDecimal("9.2");
a.divide(b <MARK>)""","""public static void main(String...aArguments) throws IOException {

    String usuario = "Jorman";
    String password = "14988611";

    String strDatos = "Jorman 14988611";
    StringTokenizer tokens = new StringTokenizer(strDatos, " ");
    int nDatos = tokens.countTokens();
    String[] datos = new String[nDatos];
    int i = 0;

    while (tokens.hasMoreTokens()) {
        String str = tokens.nextToken();
        datos[i] = str;
        i++;
    }

    //System.out.println (usuario);

    if (<MARK>) {
        System.out.println("WORKING");
    }
}"""
]
output_path = "stack-overflow.txt"
with open(output_path, "wt") as file:
    for code in samples:
        for char in code:
            if char == "\n":
                file.write("\\n")
            else:
                file.write(char)
        file.write("\n")

print(len(samples))
