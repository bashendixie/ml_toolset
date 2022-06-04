using OpenCvSharp;
using Resources;
using SmartCoalApplication.Base.StandardAnalysis;
using SmartCoalApplication.Core;
using SmartCoalApplication.Miscellaneous;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Windows.Forms;
using Point = OpenCvSharp.Point;

namespace SmartCoalApplication.Menus
{
    /// <summary>
    /// 杂项测试
    /// </summary>
    internal sealed class OpenCVMLMenu : PdnMenuItem
    {
        private PdnMenuItem menuDTrees;


        public OpenCVMLMenu()
        {
            InitializeComponent();
            //在脚本中不显示
            this.CanUseInScript = false;
            this.AutomaticScript = false;
        }

        protected override void OnAppWorkspaceChanged()
        {
            base.OnAppWorkspaceChanged();
        }

        private void InitializeComponent()
        {
            this.menuDTrees = new PdnMenuItem();
            //
            // HelpMenu
            //
            this.DropDownItems.AddRange(
                new ToolStripItem[]
                {
                    this.menuDTrees
                });

            this.Text = "OpenCV机器学习算法";
            // 
            // 视频播放/物体跟踪
            // 
            this.menuDTrees.Click += new EventHandler(this.MenuDTrees_Click);
            this.menuDTrees.Text = "决策树";

        }

        /// <summary>
        /// 读取数据集并训练数据
		///	OpenCvSharp.ML.DTrees 可以改为OpenCvSharp.ML.RTrees
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void MenuDTrees_Click1(object sender, System.EventArgs e)
        {
            //OpenCvSharp.ML.TrainData tdata = new OpenCvSharp.ML.TrainData();
            int[,] att = GetTArray(@"C:\Users\zyh\Desktop\mushroom数据集\agaricus-lepiota.train.data");
            int[] label = GetTLabel(@"C:\Users\zyh\Desktop\mushroom数据集\agaricus-lepiota.train.data");
            InputArray array = InputArray.Create(att);
            InputArray outarray = InputArray.Create(label);
            
            OpenCvSharp.ML.DTrees dtrees = OpenCvSharp.ML.DTrees.Create();
            dtrees.MaxDepth = 8;
            dtrees.MinSampleCount = 10;
            dtrees.RegressionAccuracy = 0.01f;
            dtrees.UseSurrogates = false;
            dtrees.MaxCategories = 15;
            dtrees.CVFolds = 0;
            dtrees.Use1SERule = true;
            dtrees.TruncatePrunedTree = true;
            //float[] _priors = { 1.0f, 10.0f };
            //Mat p = new Mat(1, 2, OpenCvSharp.MatType.CV_32F, _priors);
            //dtrees.Priors = p;
            dtrees.Train(array, OpenCvSharp.ML.SampleTypes.RowSample, outarray);
            //OpenCvSharp.Internal.NativeMethods.ml_
            dtrees.Save(@"C:\Users\zyh\Desktop\1.xml");
        }

		/// <summary>
        /// 读取模型并预测
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void MenuDTrees_Click(object sender, System.EventArgs e)
        {
            OpenCvSharp.ML.DTrees tree = OpenCvSharp.ML.DTrees.Load(@"C:\Users\zyh\Desktop\1.xml");
            List<int[]> att = GetTestArray(@"C:\Users\zyh\Desktop\mushroom数据集\agaricus-lepiota.test.data");

            for(int i=0;i <att.Count; i++)
            {
                Mat p = new Mat(1, 22, OpenCvSharp.MatType.CV_32F, att[i]);
                //InputArray array = InputArray.Create(att);
                //List<float> ddd = new List<float>();
                //OutputArray res = OutputArray.Create<float>(ddd);
                float rrr = tree.Predict(p);
                System.Console.WriteLine("" + rrr);
            }
        }

		//读取数据
        public int[,] GetTArray(string filepath)
        {
            //byte[] temp = System.Text.Encoding.ASCII.GetBytes("pd");
            int[,] att = new int[8000, 22];
            using (StreamReader sin = new StreamReader(new FileStream(filepath, FileMode.Open, FileAccess.Read, FileShare.Read)))
            {
                int pos = 0;
                for (string str = sin.ReadLine(); str != null; str = sin.ReadLine())
                {
                    string[] temp = str.Split(',');
                    for(int i=1; i< temp.Length; i++)
                    {
                        att[pos, i-1] = System.Text.Encoding.ASCII.GetBytes(temp[i])[0];
                    }
                    pos++;
                }
            }

            return att;
        }

		//读取标签
        public int[] GetTLabel(string filepath)
        {
            int[] att = new int[8000];
            using (StreamReader sin = new StreamReader(new FileStream(filepath, FileMode.Open, FileAccess.Read, FileShare.Read)))
            {
                int pos = 0;
                for (string str = sin.ReadLine(); str != null; str = sin.ReadLine())
                {
                    string[] temp = str.Split(',');
                    att[pos] = System.Text.Encoding.ASCII.GetBytes(temp[0])[0];
                    pos++;
                }
            }

            return att;
        }
		
		//读取测试数据
        public List<int[]> GetTestArray(string filepath)
        {
            //byte[] temp = System.Text.Encoding.ASCII.GetBytes("pd");
            List<int[]> att = new List<int[]>();
            using (StreamReader sin = new StreamReader(new FileStream(filepath, FileMode.Open, FileAccess.Read, FileShare.Read)))
            {
                for (string str = sin.ReadLine(); str != null; str = sin.ReadLine())
                {
                    string[] temp = str.Split(',');
                    int[] vs = new int[temp.Length];
                    for (int i = 1; i < temp.Length; i++)
                    {
                        vs[i-1] = System.Text.Encoding.ASCII.GetBytes(temp[i])[0];
                    }
                    att.Add(vs);
                }
            }

            return att;
        }

		











        private class MenuTitleAndLocale
        {
            public string title;
            public string locale;

            public MenuTitleAndLocale(string title, string locale)
            {
                this.title = title;
                this.locale = locale;
            }
        }

        private string GetCultureInfoName(CultureInfo ci)
        {
            CultureInfo en_US = new CultureInfo("en-US");

            // For "English (United States)" we'd rather just display "English"
            if (ci.Equals(en_US))
            {
                return GetCultureInfoName(ci.Parent);
            }
            else
            {
                return ci.NativeName;
            }
        }
    }
}
