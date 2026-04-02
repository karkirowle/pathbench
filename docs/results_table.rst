Results
=======

Speaker-level Pearson Correlation Coefficients (PCC) across all PathBench
datasets and metrics. Signs are aligned so that positive values always
indicate the expected (healthy) direction.

- **Bold**: best overall per column
- Underline: best reference-free per column
- MC: Matched Content (balanced), EX: Extended (unbalanced), Full: all data

.. raw:: html

   <style>
   .pb-results { border-collapse: collapse; font-size: 0.85em; min-width: 100%; }
   .pb-results th, .pb-results td { border: 1px solid #ddd; padding: 4px 6px; text-align: center; white-space: nowrap; }
   .pb-results th { background: #f5f5f5; }
   .pb-results tr:hover td { background: #f0f7ff; }
   .pb-results .metric-name { text-align: left; font-weight: normal; }
   .pb-results .group-header td { background: #eee; font-style: italic; text-align: left; border-bottom: 2px solid #ccc; }
   .pb-results .best-global { font-weight: bold; }
   .pb-results .best-rf { text-decoration: underline; }
   .pb-results .na { color: #999; }
   .pb-results .stat-row td { font-size: 0.9em; color: #555; }
   .pb-results .section-sep td, .pb-results .section-sep th { border-bottom: 3px double #999; }
   .pb-table-wrap { overflow-x: auto; max-width: 100%; }
   </style>
   <div class="pb-table-wrap">
   <table class="pb-results">
   <thead>
   <tr><th rowspan="3">Metric</th>
   <th colspan="2">UASpeech</th>
   <th colspan="2">NeuroVoz</th>
   <th colspan="4">EasyCall</th>
   <th colspan="6">COPAS</th>
   <th colspan="4">TORGO</th>
   <th colspan="1">YT</th>
   <th rowspan="3">Avg</th></tr>
   <tr>
   <th colspan="2">Word</th>
   <th colspan="2">Sentence</th>
   <th colspan="2">Word</th>
   <th colspan="2">Sentence</th>
   <th colspan="3">Word</th>
   <th colspan="3">Sentence</th>
   <th colspan="2">Word</th>
   <th colspan="2">Sentence</th>
   <th colspan="1">Sentence</th>
   </tr>
   <tr>
   <th>MC</th>
   <th>EX</th>
   <th>MC</th>
   <th>EX</th>
   <th>MC</th>
   <th>EX</th>
   <th>MC</th>
   <th>EX</th>
   <th>MC</th>
   <th>EX</th>
   <th>Full</th>
   <th>MC</th>
   <th>EX</th>
   <th>Full</th>
   <th>MC</th>
   <th>EX</th>
   <th>MC</th>
   <th>EX</th>
   <th>Full</th>
   </tr>
   </thead>
   <tbody>
   <tr class="group-header"><td colspan="21">Dataset Information</td></tr>
   <tr class="stat-row"><td class="metric-name">Language</td>
   <td>English</td>
   <td>English</td>
   <td>Spanish</td>
   <td>Spanish</td>
   <td>Italian</td>
   <td>Italian</td>
   <td>Italian</td>
   <td>Italian</td>
   <td>Dutch</td>
   <td>Dutch</td>
   <td>Dutch</td>
   <td>Dutch</td>
   <td>Dutch</td>
   <td>Dutch</td>
   <td>English</td>
   <td>English</td>
   <td>English</td>
   <td>English</td>
   <td>English</td>
   <td></td></tr>
   <tr class="stat-row section-sep"><td class="metric-name">Disorder</td>
   <td>Dysarthria</td>
   <td>Dysarthria</td>
   <td>Parkinson's</td>
   <td>Parkinson's</td>
   <td>Dysarthria</td>
   <td>Dysarthria</td>
   <td>Dysarthria</td>
   <td>Dysarthria</td>
   <td>Mixed pathologies</td>
   <td>Mixed pathologies</td>
   <td>Mixed pathologies</td>
   <td>Mixed pathologies</td>
   <td>Mixed pathologies</td>
   <td>Mixed pathologies</td>
   <td>Dysarthria</td>
   <td>Dysarthria</td>
   <td>Dysarthria</td>
   <td>Dysarthria</td>
   <td>Oral Cancer</td>
   <td></td></tr>
   <tr class="group-header"><td colspan="21">Statistics: Pathological</td></tr>
   <tr class="stat-row"><td class="metric-name"># Speakers</td>
   <td>14</td>
   <td>14</td>
   <td>50</td>
   <td>50</td>
   <td>30</td>
   <td>30</td>
   <td>30</td>
   <td>30</td>
   <td>11</td>
   <td>11</td>
   <td>216</td>
   <td>82</td>
   <td>82</td>
   <td>88</td>
   <td>8</td>
   <td>8</td>
   <td>8</td>
   <td>8</td>
   <td>21</td>
   <td></td></tr>
   <tr class="stat-row"><td class="metric-name"># Utterances</td>
   <td>2170</td>
   <td>6482</td>
   <td>500</td>
   <td>793</td>
   <td>1380</td>
   <td>7525</td>
   <td>600</td>
   <td>3458</td>
   <td>11</td>
   <td>532</td>
   <td>8786</td>
   <td>164</td>
   <td>164</td>
   <td>170</td>
   <td>592</td>
   <td>2046</td>
   <td>96</td>
   <td>602</td>
   <td>98</td>
   <td></td></tr>
   <tr class="group-header"><td colspan="21">Statistics: Control</td></tr>
   <tr class="stat-row"><td class="metric-name"># Speakers</td>
   <td>13</td>
   <td>13</td>
   <td>56</td>
   <td>56</td>
   <td>24</td>
   <td>24</td>
   <td>24</td>
   <td>24</td>
   <td>6</td>
   <td>130</td>
   <td>130</td>
   <td>81</td>
   <td>83</td>
   <td>83</td>
   <td>7</td>
   <td>7</td>
   <td>7</td>
   <td>7</td>
   <td>&ndash;</td>
   <td></td></tr>
   <tr class="stat-row section-sep"><td class="metric-name"># Utterances</td>
   <td>2015</td>
   <td>6045</td>
   <td>541</td>
   <td>862</td>
   <td>1104</td>
   <td>6854</td>
   <td>480</td>
   <td>3142</td>
   <td>6</td>
   <td>5967</td>
   <td>5967</td>
   <td>162</td>
   <td>164</td>
   <td>164</td>
   <td>518</td>
   <td>4097</td>
   <td>84</td>
   <td>1408</td>
   <td>&ndash;</td>
   <td></td></tr>
   <tr class="group-header"><td colspan="21">Reference-Free (Signal)</td></tr>
   <tr><td class="metric-name">Speech Rate</td>
   <td>-0.78</td>
   <td class="na">&ndash;</td>
   <td>0.26</td>
   <td>0.32</td>
   <td>-0.11</td>
   <td class="na">&ndash;</td>
   <td>0.40</td>
   <td class="na">&ndash;</td>
   <td>-0.26</td>
   <td>0.14</td>
   <td>0.12</td>
   <td>0.28</td>
   <td>0.28</td>
   <td>0.22</td>
   <td>-0.11</td>
   <td>-0.10</td>
   <td>0.72</td>
   <td>0.81</td>
   <td>0.29</td>
   <td>0.16</td>
   </tr>
   <tr><td class="metric-name">CPP</td>
   <td>-0.27</td>
   <td class="na">&ndash;</td>
   <td>0.17</td>
   <td>0.19</td>
   <td>0.16</td>
   <td class="na">&ndash;</td>
   <td>0.16</td>
   <td class="na">&ndash;</td>
   <td>-0.05</td>
   <td>-0.39</td>
   <td>0.11</td>
   <td>0.07</td>
   <td>0.07</td>
   <td>0.08</td>
   <td>-0.32</td>
   <td>-0.34</td>
   <td>-0.40</td>
   <td>-0.31</td>
   <td>-0.51</td>
   <td>-0.10</td>
   </tr>
   <tr><td class="metric-name">Std Pitch</td>
   <td>-0.36</td>
   <td class="na">&ndash;</td>
   <td>-0.30</td>
   <td>-0.31</td>
   <td>0.17</td>
   <td class="na">&ndash;</td>
   <td>0.25</td>
   <td class="na">&ndash;</td>
   <td>-0.35</td>
   <td>0.16</td>
   <td>0.08</td>
   <td>-0.03</td>
   <td>-0.03</td>
   <td>-0.01</td>
   <td>-0.23</td>
   <td>-0.30</td>
   <td>-0.36</td>
   <td>-0.47</td>
   <td>0.06</td>
   <td>-0.13</td>
   </tr>
   <tr class="group-header"><td colspan="21">Reference-Free (Speaker)</td></tr>
   <tr><td class="metric-name">VSA</td>
   <td>0.02</td>
   <td class="na">&ndash;</td>
   <td>0.01</td>
   <td>0.05</td>
   <td>0.27</td>
   <td class="na">&ndash;</td>
   <td>0.15</td>
   <td class="na">&ndash;</td>
   <td>-0.20</td>
   <td>0.55</td>
   <td>0.04</td>
   <td>0.22</td>
   <td>0.22</td>
   <td>0.20</td>
   <td class="best-global best-rf">0.63</td>
   <td class="best-rf">0.58</td>
   <td>-0.45</td>
   <td>-0.38</td>
   <td>0.17</td>
   <td>0.13</td>
   </tr>
   <tr class="group-header"><td colspan="21">Reference-Free (Model)</td></tr>
   <tr><td class="metric-name">Double ASR</td>
   <td class="best-rf">0.97</td>
   <td class="na">&ndash;</td>
   <td class="best-global best-rf">0.85</td>
   <td class="best-global best-rf">0.86</td>
   <td>0.54</td>
   <td class="na">&ndash;</td>
   <td>0.52</td>
   <td class="na">&ndash;</td>
   <td>-0.13</td>
   <td>0.28</td>
   <td>0.36</td>
   <td>0.43</td>
   <td>0.43</td>
   <td>0.46</td>
   <td>0.45</td>
   <td>0.44</td>
   <td>0.88</td>
   <td>0.87</td>
   <td>0.55</td>
   <td>0.55</td>
   </tr>
   <tr><td class="metric-name">DArtP</td>
   <td class="best-rf">0.97</td>
   <td class="na">&ndash;</td>
   <td>0.78</td>
   <td>0.78</td>
   <td class="best-rf">0.69</td>
   <td class="na">&ndash;</td>
   <td class="best-rf">0.62</td>
   <td class="na">&ndash;</td>
   <td class="best-global best-rf">0.48</td>
   <td>0.60</td>
   <td>0.50</td>
   <td>0.36</td>
   <td>0.36</td>
   <td>0.34</td>
   <td>0.54</td>
   <td>0.57</td>
   <td class="best-rf">0.92</td>
   <td class="best-rf">0.91</td>
   <td>0.78</td>
   <td class="best-rf">0.64</td>
   </tr>
   <tr><td class="metric-name">Confidence</td>
   <td>0.93</td>
   <td class="na">&ndash;</td>
   <td>0.76</td>
   <td>0.78</td>
   <td>0.60</td>
   <td class="na">&ndash;</td>
   <td>0.59</td>
   <td class="na">&ndash;</td>
   <td>0.13</td>
   <td class="best-rf">0.76</td>
   <td class="best-rf">0.54</td>
   <td class="best-rf">0.48</td>
   <td class="best-rf">0.48</td>
   <td class="best-rf">0.49</td>
   <td>0.27</td>
   <td>0.35</td>
   <td>0.90</td>
   <td>0.89</td>
   <td class="best-global best-rf">0.79</td>
   <td>0.61</td>
   </tr>
   <tr class="group-header"><td colspan="21">Reference-Text</td></tr>
   <tr><td class="metric-name">PER (SEM)</td>
   <td>0.84</td>
   <td class="na">&ndash;</td>
   <td>0.83</td>
   <td>0.85</td>
   <td>0.41</td>
   <td class="na">&ndash;</td>
   <td>0.61</td>
   <td class="na">&ndash;</td>
   <td>0.30</td>
   <td>0.54</td>
   <td>0.49</td>
   <td>0.28</td>
   <td>0.28</td>
   <td>0.28</td>
   <td>0.48</td>
   <td>0.46</td>
   <td class="best-global">0.93</td>
   <td>0.91</td>
   <td>0.66</td>
   <td>0.57</td>
   </tr>
   <tr><td class="metric-name">PER (Phone)</td>
   <td>0.75</td>
   <td class="na">&ndash;</td>
   <td>0.70</td>
   <td>0.72</td>
   <td>0.46</td>
   <td class="na">&ndash;</td>
   <td>0.55</td>
   <td class="na">&ndash;</td>
   <td>0.26</td>
   <td>0.61</td>
   <td>0.55</td>
   <td>0.18</td>
   <td>0.18</td>
   <td>0.17</td>
   <td>0.49</td>
   <td>0.59</td>
   <td>0.92</td>
   <td class="best-global">0.92</td>
   <td>0.75</td>
   <td>0.55</td>
   </tr>
   <tr><td class="metric-name">ArtP</td>
   <td class="best-global">0.98</td>
   <td class="na">&ndash;</td>
   <td>0.77</td>
   <td>0.78</td>
   <td>0.70</td>
   <td class="na">&ndash;</td>
   <td>0.68</td>
   <td class="na">&ndash;</td>
   <td>0.40</td>
   <td class="best-global">0.78</td>
   <td class="best-global">0.61</td>
   <td>0.57</td>
   <td>0.57</td>
   <td>0.56</td>
   <td>0.56</td>
   <td class="best-global">0.61</td>
   <td>0.93</td>
   <td class="best-global">0.92</td>
   <td>0.78</td>
   <td class="best-global">0.70</td>
   </tr>
   <tr class="group-header"><td colspan="21">Reference-Audio (Parallel)</td></tr>
   <tr><td class="metric-name">P-ESTOI</td>
   <td>0.96</td>
   <td class="na">&ndash;</td>
   <td>0.40</td>
   <td>0.53</td>
   <td>0.57</td>
   <td class="na">&ndash;</td>
   <td>0.70</td>
   <td class="na">&ndash;</td>
   <td>0.23</td>
   <td>0.40</td>
   <td>0.39</td>
   <td>0.42</td>
   <td>0.41</td>
   <td>0.42</td>
   <td>0.43</td>
   <td>0.40</td>
   <td>0.76</td>
   <td>0.86</td>
   <td class="na">&ndash;</td>
   <td>0.53</td>
   </tr>
   <tr><td class="metric-name">NAD</td>
   <td>0.97</td>
   <td class="na">&ndash;</td>
   <td>0.70</td>
   <td>0.75</td>
   <td class="best-global">0.75</td>
   <td class="na">&ndash;</td>
   <td class="best-global">0.78</td>
   <td class="na">&ndash;</td>
   <td>0.18</td>
   <td>0.46</td>
   <td>0.51</td>
   <td class="best-global">0.69</td>
   <td class="best-global">0.69</td>
   <td class="best-global">0.70</td>
   <td>0.52</td>
   <td>0.55</td>
   <td>0.90</td>
   <td>0.90</td>
   <td class="na">&ndash;</td>
   <td>0.67</td>
   </tr>
   </tbody>
   </table>
   </div>
