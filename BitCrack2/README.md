# BitCrack2
_Hunt for Bitcoin private keys._

It is a modified version of [BitCrack](https://github.com/brichard19/BitCrack) by [brichard19](https://github.com/brichard19),
for random and xpoint mode.

Thank you him for his hardwork.

## Note: It's a experimental project.
# Changes 

- Added random stride option, by using this option, the program run indefinitely after the end of keyspace, it starts again from starting range with the updated random stride of given bit length, it continues doing this until it found keys or you stoped it. This is like a random walk with random distance.
- For --rstride N, N should be ```128 >= N >= 1```, here N is the bit length of the incrementor.
- Added support for XPoint search mode.
- Saving-Loading checkpoints are not modified for new changes.
- OpenCL support removed.
- Everything else is same as original BitCrack.
  
# Usage

- For XPoint mode use x point of the public key, without ```02``` or ```03``` prefix(64 chars).
- Don't use XPoint mode with 'uncompressed' compression type.
- Address or XPoint file should be in text format with one address or xpoint per line.

```
BitCrack.exe --help
BitCrack OPTIONS [TARGETS]
Where TARGETS is one or more addresses

--help                       Display this message
-c, --compressed             Use compressed points
-u, --uncompressed           Use Uncompressed points
--compression  MODE          Specify compression where MODE is
                                 COMPRESSED or UNCOMPRESSED or BOTH
-d, --device ID              Use device ID
-b, --blocks N               N blocks
-t, --threads N              N threads per block
-p, --points N               N points per thread
-i, --in FILE                Read addresses from FILE, one per line
-o, --out FILE               Write keys to FILE
-f, --follow                 Follow text output
-m, --mode MODE              Specify search mode where MODE is
                                 ADDRESS or XPOINT
--list-devices               List available devices
--keyspace KEYSPACE          Specify the keyspace:
                                 START:END
                                 START:+COUNT
                                 START
                                 :END
                                 :+COUNT
                             Where START, END, COUNT are in hex format
--stride N                   Increment by N keys at a time
--rstride N                  Random stride bits[1 to 128], continue after end of range by setting up new random stride
--share M/N                  Divide the keyspace into N equal shares, process the Mth share
--continue FILE              Save/load progress from FILE
-v, --version                Show version
```

# Address Search Mode 
For puzzle ```35```, ```36``` and ```37``` with ```--rstride``` of ```5``` bit
```
BitCrack.exe -b 64 -t 256 -p 1024 --rstride 5 --keyspace 400000000:1FFFFFFFFF 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb 1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1 14iXhn8bGajVWegZHJ18vJLHhntcpL4dex
[2021-07-14.17:01:04] [Info] Compression : compressed
[2021-07-14.17:01:04] [Info] Starting at : 400000000 (35 bit)
[2021-07-14.17:01:04] [Info] Ending at   : 1FFFFFFFFF (37 bit)
[2021-07-14.17:01:04] [Info] Range       : 1BFFFFFFFF (37 bit)
[2021-07-14.17:01:04] [Info] Initializing GeForce GTX 1650
[2021-07-14.17:01:04] [Info] Generating 16,777,216 starting points (640.0MB)
[2021-07-14.17:01:09] [Info] 10.0%  20.0%  30.0%  40.0%  50.0%  60.0%  70.0%  80.0%  90.0%  100.0%
[2021-07-14.17:01:11] [Info] Done
[DEV: GeForce GTX 1650 3334/4096MB] [K: 78E000000 (35 bit), C: 12.695313 %] [I: 1A (5 bit), 3] [T: 3] [S: 306.85 MK/s] [12,096,372,736 (34 bit)] [00:00:50]
[2021-07-14.17:02:05] [Info] Found key for address '1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1'. Written to 'Found.txt'
[2021-07-14.17:02:05] [Info] Address     : 1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1
                             Private key : 9DE820A7C
                             Compressed  : yes
                             Public key  : 02B3E772216695845FA9DDA419FB5DACA28154D8AA59EA302F05E916635E47B9F6

[DEV: GeForce GTX 1650 3334/4096MB] [K: 418000000 (35 bit), C: 0.334821 %] [I: 18 (5 bit), 12] [T: 2] [S: 30.74 MK/s] [57,327,747,072 (36 bit)] [00:04:13] 1]
[2021-07-14.17:05:31] [Info] Found key for address '1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb'. Written to 'Found.txt'
[2021-07-14.17:05:31] [Info] Address     : 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb
                             Private key : 4AED21170
                             Compressed  : yes
                             Public key  : 02F6A8148A62320E149CB15C544FE8A25AB483A0095D2280D03B8A00A7FEADA13D

[DEV: GeForce GTX 1650 3334/4096MB] [K: 1497000000 (37 bit), C: 59.249442 %] [I: 1F (5 bit), 17] [T: 1] [S: 306.85 MK/s] [86,939,533,312 (37 bit)] [00:06:27]
[2021-07-14.17:07:41] [Info] Found key for address '14iXhn8bGajVWegZHJ18vJLHhntcpL4dex'. Written to 'Found.txt'
[2021-07-14.17:07:41] [Info] Address     : 14iXhn8bGajVWegZHJ18vJLHhntcpL4dex
                             Private key : 1757756A93
                             Compressed  : yes
                             Public key  : 027D2C03C3EF0AEC70F2C7E1E75454A5DFDD0E1ADEA670C1B3A4643C48AD0F1255


[2021-07-14.17:07:41] [Info] No targets remaining
```

# XPoint Search Mode 
For puzzle ```35```, ```36``` and ```37``` with ```--rstride``` of ```5``` bit
```
BitCrack.exe -b 64 -t 256 -p 1024 --rstride 5 -m xpoint --keyspace 400000000:1fffffffff f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d b3e772216695845fa9dda419fb5daca28154d8aa59ea302f05e916635e47b9f6 7d2c03c3ef0aec70f2c7e1e75454a5dfdd0e1adea670c1b3a4643c48ad0f1255
[2021-07-18.17:31:54] [Info] Compression : compressed
[2021-07-18.17:31:54] [Info] Seach mode  : XPOINT
[2021-07-18.17:31:54] [Info] Starting at : 400000000 (35 bit)
[2021-07-18.17:31:54] [Info] Ending at   : 1FFFFFFFFF (37 bit)
[2021-07-18.17:31:54] [Info] Range       : 1BFFFFFFFF (37 bit)
[2021-07-18.17:31:54] [Info] Initializing GeForce GTX 1650
[2021-07-18.17:31:55] [Info] Generating 16,777,216 starting points (640.0MB)
[2021-07-18.17:31:59] [Info] 10.0%  20.0%  30.0%  40.0%  50.0%  60.0%  70.0%  80.0%  90.0%  100.0%
[2021-07-18.17:32:01] [Info] Done
[DEV: GeForce GTX 1650 3334/4096MB] [K: 962000000 (36 bit), C: 19.224330 %] [I: 1A (5 bit), 3] [T: 3] [S: 477.25 MK/s] [10,787,749,888 (34 bit)] [00:00:33]
[2021-07-18.17:32:36] [Info] Found key for address '1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1'. Written to 'Found.txt'
[2021-07-18.17:32:36] [Info] Address     : 1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1
                             Private key : 9DE820A7C
                             Compressed  : yes
                             Public key  : 02B3E772216695845FA9DDA419FB5DACA28154D8AA59EA302F05E916635E47B9F6

[DEV: GeForce GTX 1650 3334/4096MB] [K: 1500000000 (37 bit), C: 60.714286 %] [I: 11 (5 bit), 5] [T: 2] [S: 471.95 MK/s] [24,847,056,896 (35 bit)] [00:01:15]
[2021-07-18.17:33:19] [Info] Found key for address '14iXhn8bGajVWegZHJ18vJLHhntcpL4dex'. Written to 'Found.txt'
[2021-07-18.17:33:19] [Info] Address     : 14iXhn8bGajVWegZHJ18vJLHhntcpL4dex
                             Private key : 1757756A93
                             Compressed  : yes
                             Public key  : 027D2C03C3EF0AEC70F2C7E1E75454A5DFDD0E1ADEA670C1B3A4643C48AD0F1255

[DEV: GeForce GTX 1650 3334/4096MB] [K: 418000000 (35 bit), C: 0.334821 %] [I: 18 (5 bit), 14] [T: 1] [S: 27.28 MK/s] [69,189,238,784 (37 bit)] [00:03:42] 0]
[2021-07-18.17:35:50] [Info] Found key for address '1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb'. Written to 'Found.txt'
[2021-07-18.17:35:50] [Info] Address     : 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb
                             Private key : 4AED21170
                             Compressed  : yes
                             Public key  : 02F6A8148A62320E149CB15C544FE8A25AB483A0095D2280D03B8A00A7FEADA13D


[2021-07-18.17:35:50] [Info] No targets remaining
```

## Attempting puzzle ```160``` with ```--rstride``` of ```124``` bit in xpoint search mode
```
BitCrack.exe -b 64 -t 256 -p 1024 --rstride 124 --mode xpoint --keyspace 8000000000000000000000000000000000000000:ffffffffffffffffffffffffffffffffffffffff e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673
[2021-07-20.00:33:10] [Info] Compression : compressed
[2021-07-20.00:33:10] [Info] Seach mode  : XPOINT
[2021-07-20.00:33:10] [Info] Starting at : 8000000000000000000000000000000000000000 (160 bit)
[2021-07-20.00:33:10] [Info] Ending at   : FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF (160 bit)
[2021-07-20.00:33:10] [Info] Range       : 7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF (159 bit)
[2021-07-20.00:33:10] [Info] RStride     : 124 bit
[2021-07-20.00:33:10] [Info] Initializing GeForce GTX 1650
[2021-07-20.00:33:10] [Info] Generating 16,777,216 starting points (640.0MB)
[2021-07-20.00:33:15] [Info] 10.0%  20.0%  30.0%  40.0%  50.0%  60.0%  70.0%  80.0%  90.0%  100.0%
[2021-07-20.00:33:25] [Info] Done
[DEV: GeForce GTX 1650 3435/4096MB] [K: 9CA2365A12561A658624835BA2ADC0CB68000000 (160 bit), C: 22.370033 %] [I: A93D77EBBC8A07840546A2705CAB6C8 (124 bit), 9] [T: 1] [S: 453.69 MK/s] [384,886,112,256 (39 bit)] [00:16:07]
```

## with xpoints file:
For puzzle ```1``` to ```37``` with ```--stride``` of ```1```
```
BitCrack.exe -b 64 -t 256 -p 1024 --stride 1 -m xpoint --keyspace 1:1fffffffff -i "puzzle-1-37-xpoints.txt"
[2021-07-18.16:57:07] [Info] Compression : compressed
[2021-07-18.16:57:07] [Info] Seach type  : XPOINT
[2021-07-18.16:57:07] [Info] Starting at : 1 (1 bit)
[2021-07-18.16:57:07] [Info] Ending at   : 1FFFFFFFFF (37 bit)
[2021-07-18.16:57:07] [Info] Range       : 1FFFFFFFFE (37 bit)
[2021-07-18.16:57:07] [Info] Initializing GeForce GTX 1650
[2021-07-18.16:57:07] [Info] Generating 16,777,216 starting points (640.0MB)
[2021-07-18.16:57:12] [Info] 10.0%  20.0%  30.0%  40.0%  50.0%  60.0%  70.0%  80.0%  90.0%  100.0%
[2021-07-18.16:57:14] [Info] Done
[2021-07-18.16:57:14] [Info] Loading xpoints from 'G:\BTCPUBKEYS\puzzle-1-37-pubkeys.txt'
[2021-07-18.16:57:14] [Info] 37 xpoints loaded (0.0MB)
[2021-07-18.16:57:14] [Info] Allocating bloom filter (0.0MB)

[2021-07-18.16:57:14] [Info] Found key for address '1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH
                             Private key : 1
                             Compressed  : yes
                             Public key  : 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798


[2021-07-18.16:57:14] [Info] Found key for address '1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb
                             Private key : 3
                             Compressed  : yes
                             Public key  : 02F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9


[2021-07-18.16:57:14] [Info] Found key for address '19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA
                             Private key : 7
                             Compressed  : yes
                             Public key  : 025CBDF0646E5DB4EAA398F365F2EA7A0E3D419B7E0330E39CE92BDDEDCAC4F9BC


[2021-07-18.16:57:14] [Info] Found key for address '1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e
                             Private key : 8
                             Compressed  : yes
                             Public key  : 022F01E5E15CCA351DAFF3843FB70F3C2F0A1BDD05E5AF888A67784EF3E10A2A01


[2021-07-18.16:57:14] [Info] Found key for address '1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k
                             Private key : 15
                             Compressed  : yes
                             Public key  : 02352BBF4A4CDD12564F93FA332CE333301D9AD40271F8107181340AEF25BE59D5


[2021-07-18.16:57:14] [Info] Found key for address '1PitScNLyp2HCygzadCh7FveTnfmpPbfp8'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1PitScNLyp2HCygzadCh7FveTnfmpPbfp8
                             Private key : 31
                             Compressed  : yes
                             Public key  : 03F2DAC991CC4CE4B9EA44887E5C7C0BCE58C80074AB9D4DBAEB28531B7739F530


[2021-07-18.16:57:14] [Info] Found key for address '1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK
                             Private key : E0
                             Compressed  : yes
                             Public key  : 038BC89C2F919ED158885C35600844D49890905C79B357322609C45706CE6B514


[2021-07-18.16:57:14] [Info] Found key for address '1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe
                             Private key : 202
                             Compressed  : yes
                             Public key  : 03A7A4C30291AC1DB24B4AB00C442AA832F7794B5A0959BEC6E8D7FEE802289DCD


[2021-07-18.16:57:14] [Info] Found key for address '1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu
                             Private key : 483
                             Compressed  : yes
                             Public key  : 038B05B0603ABD75B0C57489E451F811E1AFE54A8715045CDF4888333F3EBC6E8B


[2021-07-18.16:57:14] [Info] Found key for address '1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1
                             Private key : 1460
                             Compressed  : yes
                             Public key  : 03AADAAAB1DB8D5D450B511789C37E7CFEB0EB8B3E61A57A34166C5EDC9A4B869D


[2021-07-18.16:57:14] [Info] Found key for address '1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC
                             Private key : 4C
                             Compressed  : yes
                             Public key  : 0296516A8F65774275278D0D7420A88DF0AC44BD64C7BAE07C3FE397C5B3300B23


[2021-07-18.16:57:14] [Info] Found key for address '1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV
                             Private key : 1D3
                             Compressed  : yes
                             Public key  : 0243601D61C836387485E9514AB5C8924DD2CFD466AF34AC95002727E1659D60F7


[2021-07-18.16:57:14] [Info] Found key for address '1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot
                             Private key : A7B
                             Compressed  : yes
                             Public key  : 038B00FCBFC1A203F44BF123FC7F4C91C10A85C8EAE9187F9D22242B4600CE781C


[2021-07-18.16:57:14] [Info] Found key for address '1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY
                             Private key : C936
                             Compressed  : yes
                             Public key  : 029D8C5D35231D75EB87FD2C5F05F65281ED9573DC41853288C62EE94EB2590B7A


[2021-07-18.16:57:14] [Info] Found key for address '1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE
                             Private key : 3080D
                             Compressed  : yes
                             Public key  : 02CE4A3291B19D2E1A7BF73EE87D30A6BDBC72B20771E7DFFF40D0DB755CD4AF1


[2021-07-18.16:57:14] [Info] Found key for address '14oFNXucftsHiUMY8uctg6N487riuyXs4h'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 14oFNXucftsHiUMY8uctg6N487riuyXs4h
                             Private key : 1BA534
                             Compressed  : yes
                             Public key  : 031A746C78F72754E0BE046186DF8A20CDCE5C79B2EDA76013C647AF08D306E49E


[2021-07-18.16:57:14] [Info] Found key for address '1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW
                             Private key : 68F3
                             Compressed  : yes
                             Public key  : 02FEA58FFCF49566F6E9E9350CF5BCA2861312F422966E8DB16094BEB14DC3DF2C


[2021-07-18.16:57:14] [Info] Found key for address '1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk
                             Private key : 2930
                             Compressed  : yes
                             Public key  : 03B4F1DE58B8B41AFE9FD4E5FFBDAFAEAB86C5DB4769C15D6E6011AE7351E54759


[2021-07-18.16:57:14] [Info] Found key for address '1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv
                             Private key : 2DE40F
                             Compressed  : yes
                             Public key  : 023ED96B524DB5FF4FE007CE730366052B7C511DC566227D929070B9CE917ABB43


[2021-07-18.16:57:14] [Info] Found key for address '1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum
                             Private key : D2C55
                             Compressed  : yes
                             Public key  : 033C4A45CBD643FF97D77F41EA37E843648D50FD894B864B0D52FEBC62F6454F7C


[2021-07-18.16:57:14] [Info] Found key for address '1L2GM8eE7mJWLdo3HZS6su1832NX2txaac'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1L2GM8eE7mJWLdo3HZS6su1832NX2txaac
                             Private key : 556E52
                             Compressed  : yes
                             Public key  : 03F82710361B8B81BDEDB16994F30C80DB522450A93E8E87EEB07F7903CF28D04B


[2021-07-18.16:57:14] [Info] Found key for address '1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7
                             Private key : DC2A04
                             Compressed  : yes
                             Public key  : 036EA839D22847EE1DCE3BFC5B11F6CF785B0682DB58C35B63D1342EB221C3490C


[2021-07-18.16:57:14] [Info] Found key for address '1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w
                             Private key : 5749F
                             Compressed  : yes
                             Public key  : 0385663C8B2F90659E1CCAB201694F4F8EC24B3749CFE5030C7C3646A709408E19


[2021-07-18.16:57:14] [Info] Found key for address '1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm
                             Private key : 1764F
                             Compressed  : yes
                             Public key  : 033F688BAE8321B8E02B7E6C0A55C2515FB25AB97D85FDA842449F7BFA04E128C3


[2021-07-18.16:57:14] [Info] Found key for address '15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP
                             Private key : 1FA5EE5
                             Compressed  : yes
                             Public key  : 0357FBEA3A2623382628DDE556B2A0698E32428D3CD225F3BD034DCA82DD7455A


[2021-07-18.16:57:14] [Info] Found key for address '1JVnST957hGztonaWK6FougdtjxzHzRMMg'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 1JVnST957hGztonaWK6FougdtjxzHzRMMg
                             Private key : 340326E
                             Compressed  : yes
                             Public key  : 024E4F50A2A3ECCDB368988AE37CD4B611697B26B29696E42E06D71368B4F3840F


[2021-07-18.16:57:14] [Info] Found key for address '128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k
                             Private key : 6AC3875
                             Compressed  : yes
                             Public key  : 031A864BAE3922F351F1B57CFDD827C25B7E093CB9C88A72C1CD893D9F90F44ECE


[2021-07-18.16:57:14] [Info] Found key for address '12jbtzBb54r97TCwW3G1gCFoumpckRAPdY'. Written to 'Found.txt'
[2021-07-18.16:57:14] [Info] Address     : 12jbtzBb54r97TCwW3G1gCFoumpckRAPdY
                             Private key : D916CE8
                             Compressed  : yes
                             Public key  : 03E9E661838A96A65331637E2A3E948DC0756E5009E7CB5C36664D9B72DD18C0A7


[2021-07-18.16:57:15] [Info] Found key for address '19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT'. Written to 'Found.txt'
[2021-07-18.16:57:15] [Info] Address     : 19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT
                             Private key : 17E2551E
                             Compressed  : yes
                             Public key  : 026CAAD634382D34691E3BEF43ED4A124D8909A8A3362F91F1D20ABAAF7E917B36

[DEV: GeForce GTX 1650 3334/4096MB] [K: 2D000001 (30 bit), C: 0.549316 %] [I: 1 (1 bit), 1] [T: 8] [S: 416.65 MK/s] [754,974,720 (30 bit)] [00:00:00]
[2021-07-18.16:57:16] [Info] Found key for address '1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps'. Written to 'Found.txt'
[2021-07-18.16:57:16] [Info] Address     : 1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps
                             Private key : 3D94CD64
                             Compressed  : yes
                             Public key  : 03D282CF2FF536D2C42F105D0B8588821A915DC3F9A05BD98BB23AF67A2E92A5B

[DEV: GeForce GTX 1650 3334/4096MB] [K: 60000001 (31 bit), C: 1.171875 %] [I: 1 (1 bit), 1] [T: 7] [S: 468.07 MK/s] [1,610,612,736 (31 bit)] [00:00:01]
[2021-07-18.16:57:18] [Info] Found key for address '1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE'. Written to 'Found.txt'
[2021-07-18.16:57:18] [Info] Address     : 1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE
                             Private key : 7D4FE747
                             Compressed  : yes
                             Public key  : 0387DC70DB1806CD9A9A76637412EC11DD998BE666584849B3185F7F9313C8FD28

[DEV: GeForce GTX 1650 3334/4096MB] [K: 93000001 (32 bit), C: 1.794434 %] [I: 1 (1 bit), 1] [T: 6] [S: 468.07 MK/s] [2,466,250,752 (32 bit)] [00:00:03]
[2021-07-18.16:57:21] [Info] Found key for address '1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR'. Written to 'Found.txt'
[2021-07-18.16:57:21] [Info] Address     : 1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR
                             Private key : B862A62E
                             Compressed  : yes
                             Public key  : 029C58240E50E3BA3F833C82655E8725C037A2294E14CF5D73A5DF8D56159DE69

[DEV: GeForce GTX 1650 3334/4096MB] [K: 18D000001 (33 bit), C: 4.846191 %] [I: 1 (1 bit), 1] [T: 5] [S: 436.72 MK/s] [6,660,554,752 (33 bit)] [00:00:12]
[2021-07-18.16:57:29] [Info] Found key for address '187swFMjz1G54ycVU56B7jZFHFTNVQFDiu'. Written to 'Found.txt'
[2021-07-18.16:57:29] [Info] Address     : 187swFMjz1G54ycVU56B7jZFHFTNVQFDiu
                             Private key : 1A96CA8D8
                             Compressed  : yes
                             Public key  : 03A355AA5E2E09DD44BB46A4722E9336E9E3EE4EE4E7B7A0CF5785B283BF2AB579

[DEV: GeForce GTX 1650 3334/4096MB] [K: 322000001 (34 bit), C: 9.790039 %] [I: 1 (1 bit), 1] [T: 4] [S: 458.90 MK/s] [13,455,327,232 (34 bit)] [00:00:27]
[2021-07-18.16:57:44] [Info] Found key for address '1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q'. Written to 'Found.txt'
[2021-07-18.16:57:44] [Info] Address     : 1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q
                             Private key : 34A65911D
                             Compressed  : yes
                             Public key  : 033CDD9D6D97CBFE7C26F902FAF6A435780FE652E159EC953650EC7B1004082790

[DEV: GeForce GTX 1650 3334/4096MB] [K: 485000001 (35 bit), C: 14.123535 %] [I: 1 (1 bit), 1] [T: 3] [S: 472.21 MK/s] [19,411,238,912 (35 bit)] [00:00:40]
[2021-07-18.16:57:57] [Info] Found key for address '1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb'. Written to 'Found.txt'
[2021-07-18.16:57:57] [Info] Address     : 1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb
                             Private key : 4AED21170
                             Compressed  : yes
                             Public key  : 02F6A8148A62320E149CB15C544FE8A25AB483A0095D2280D03B8A00A7FEADA13D

[DEV: GeForce GTX 1650 3334/4096MB] [K: 9DB000001 (36 bit), C: 30.798340 %] [I: 1 (1 bit), 1] [T: 2] [S: 472.21 MK/s] [42,328,915,968 (36 bit)] [00:01:29]
[2021-07-18.16:58:45] [Info] Found key for address '1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1'. Written to 'Found.txt'
[2021-07-18.16:58:45] [Info] Address     : 1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1
                             Private key : 9DE820A7C
                             Compressed  : yes
                             Public key  : 02B3E772216695845FA9DDA419FB5DACA28154D8AA59EA302F05E916635E47B9F6

[DEV: GeForce GTX 1650 3334/4096MB] [K: 173C000001 (37 bit), C: 72.607422 %] [I: 1 (1 bit), 1] [T: 1] [S: 473.37 MK/s] [99,790,880,768 (37 bit)] [00:03:31]
[2021-07-18.17:00:48] [Info] Found key for address '14iXhn8bGajVWegZHJ18vJLHhntcpL4dex'. Written to 'Found.txt'
[2021-07-18.17:00:48] [Info] Address     : 14iXhn8bGajVWegZHJ18vJLHhntcpL4dex
                             Private key : 1757756A93
                             Compressed  : yes
                             Public key  : 027D2C03C3EF0AEC70F2C7E1E75454A5DFDD0E1ADEA670C1B3A4643C48AD0F1255


[2021-07-18.17:00:48] [Info] No targets remaining
```



# Building
### Windows
- Microsoft Visual Studio Community 2019
- CUDA version 10.0
## Linux
- Install libgmp: ```sudo apt install -y libgmp-dev```

- Edit the makefile and set up the appropriate CUDA SDK and compiler paths for nvcc.

    ```make
    CUDA       = /usr/local/cuda-11.0
    CXXCUDA    = /usr/bin/g++
    ```
 - To build with CUDA: pass CCAP value according to your GPU compute capability
    ```sh
    $ cd BitCrack
    $ make CCAP=75 all
    ```

# License
BitCrack is licensed under MIT License.

## Donation
- BTC: bc1qwnd4jqe0ulldr7rn8dk8mpkytm7jz068mpau9w
- ETH: 0x16DD2B876C91d168cBAF7c18b78b9d0aDc011365

## __Disclaimer__
ALL THE CODES, PROGRAM AND INFORMATION ARE FOR EDUCATIONAL PURPOSES ONLY. USE IT AT YOUR OWN RISK. THE DEVELOPER WILL NOT BE RESPONSIBLE FOR ANY LOSS, DAMAGE OR CLAIM ARISING FROM USING THIS PROGRAM.