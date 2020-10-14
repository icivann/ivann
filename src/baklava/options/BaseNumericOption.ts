import { Prop, Vue, Component } from 'vue-property-decorator';

@Component
export default class BaseNumericOption extends Vue {
  @Prop()
  value!: any;

  @Prop({ type: String })
  name!: string;

  get v() {
    if (typeof this.value === 'string') {
      return parseFloat(this.value);
    }
    if (typeof this.value === 'number') {
      return this.value;
    }
    return 0;
  }

  setValue(newValue: number) {
    this.$emit('input', newValue);
    this.value = newValue;
    console.log(newValue);
  }
}
